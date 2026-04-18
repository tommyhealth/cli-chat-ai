import hashlib
import json
import os
import re
from typing import List, Dict, Optional

import chromadb
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from rank_bm25 import BM25Okapi
import spacy
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

load_dotenv()

# ---- Маппинг римских цифр ----
ROMAN_TO_INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
}
PART_NAME_TO_NUM = {
    "первая": 1, "вторая": 2, "третья": 3, "четвертая": 4,
    "пятая": 5, "шестая": 6,
}



# =============================================================================
# 1. StructureParser
# =============================================================================

class StructureParser:
    """Парсит текст книги в структурированный список частей и глав."""

    PART_PATTERN = re.compile(
        r"^Часть\s+(первая|вторая|третья|четвертая|пятая|шестая)$", re.IGNORECASE
    )
    CHAPTER_PATTERN_ROMAN = re.compile(r"^([IVX]+)$")
    CHAPTER_PATTERN_ARABIC = re.compile(r"^(\d+)\.\s+.+$")

    def parse(self, text: str) -> List[Dict]:
        lines = text.split("\n")
        sections = []
        current_part = 0
        current_chapter = 0
        current_lines: List[str] = []

        for line in lines:
            stripped = line.strip()

            part_match = self.PART_PATTERN.match(stripped)
            if part_match:
                if current_lines and current_chapter > 0:
                    sections.append({
                        "part": current_part,
                        "chapter": current_chapter,
                        "text": "\n".join(current_lines).strip(),
                    })
                    current_lines = []
                current_part = PART_NAME_TO_NUM[part_match.group(1).lower()]
                current_chapter = 0
                continue

            chapter_match_roman = self.CHAPTER_PATTERN_ROMAN.match(stripped)
            if chapter_match_roman and stripped in ROMAN_TO_INT:
                if current_lines and current_chapter > 0:
                    sections.append({
                        "part": current_part,
                        "chapter": current_chapter,
                        "text": "\n".join(current_lines).strip(),
                    })
                    current_lines = []
                current_chapter = ROMAN_TO_INT[stripped]
                continue

            chapter_match_arabic = self.CHAPTER_PATTERN_ARABIC.match(stripped)
            if chapter_match_arabic and current_part > 0:
                if current_lines and current_chapter > 0:
                    sections.append({
                        "part": current_part,
                        "chapter": current_chapter,
                        "text": "\n".join(current_lines).strip(),
                    })
                    current_lines = []
                current_chapter = int(chapter_match_arabic.group(1))
                continue

            if current_part > 0 and current_chapter > 0:
                current_lines.append(line)

        if current_lines and current_chapter > 0:
            sections.append({
                "part": current_part,
                "chapter": current_chapter,
                "text": "\n".join(current_lines).strip(),
            })

        return sections


# =============================================================================
# 2. BookSummarizer — Map-Reduce суммаризация
# =============================================================================

class BookSummarizer:
    """Генерирует и кэширует суммаризацию книги через map-reduce подход."""

    CACHE_FILE = "book_summaries.json"
    CHUNK_SIZE = 3000  # макс. символов для одного LLM-вызова в map-фазе

    def __init__(self):
        self.credentials = os.environ.get("GIGACHAT_CREDENTIALS")
        if not self.credentials:
            raise RuntimeError("Установите переменную GIGACHAT_CREDENTIALS")

    def get_or_create(self, book_path: str, sections: List[Dict]) -> Dict:
        cache_path = os.path.join(os.path.dirname(book_path), self.CACHE_FILE)
        book_hash = self._hash_file(book_path)

        # Проверяем кэш
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("book_hash") == book_hash:
                print("   Суммаризация загружена из кэша")
                return cache

        # Генерируем
        print("   Кэш не найден или устарел — генерирую суммаризацию...")
        chapter_summaries = self._map_phase(sections)
        book_summary = self._reduce_phase(chapter_summaries)
        character_aliases = self._extract_aliases(chapter_summaries)

        result = {
            "book_hash": book_hash,
            "chapter_summaries": chapter_summaries,
            "book_summary": book_summary,
            "character_aliases": character_aliases,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"   Суммаризация сохранена в {cache_path}")

        return result

    def _hash_file(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _map_phase(self, sections: List[Dict]) -> List[Dict]:
        """Map: каждая глава -> краткое содержание (2-3 предложения)."""
        chapter_summaries = []

        for i, section in enumerate(sections):
            part = section["part"]
            chapter = section["chapter"]
            text = section["text"]

            print(f"   Map: Часть {part}, Глава {chapter} ({len(text)} символов)...")

            summary = self._summarize_chapter(text)

            chapter_summaries.append({
                "part": part,
                "chapter": chapter,
                "summary": summary,
            })

        return chapter_summaries

    def _summarize_chapter(self, text: str) -> str:
        """Суммаризация одной главы с обработкой длинных текстов."""
        if len(text) <= self.CHUNK_SIZE * 1.5:
            # Глава достаточно короткая — суммаризуем целиком
            return self._call_llm_summarize(
                text,
                "Кратко опиши содержание этой главы книги в 2-3 предложениях. "
                "Укажи ключевых персонажей и основные события."
            )

        # Длинная глава — разбиваем на куски, суммаризуем каждый, потом объединяем
        chunks = self._split_text(text, self.CHUNK_SIZE)
        chunk_summaries = []

        for j, chunk in enumerate(chunks):
            s = self._call_llm_summarize(
                chunk,
                "Кратко опиши содержание этого фрагмента главы в 1-2 предложениях. "
                "Укажи ключевых персонажей и события."
            )
            chunk_summaries.append(s)

        # Объединяем суммари кусков в одно суммари главы
        combined = "\n".join(f"- {s}" for s in chunk_summaries)
        return self._call_llm_summarize(
            combined,
            "На основе этих кратких описаний частей одной главы, "
            "составь единое краткое содержание главы в 2-3 предложениях. "
            "Укажи ключевых персонажей и основные события."
        )

    def _reduce_phase(self, chapter_summaries: List[Dict]) -> str:
        """Reduce: все саммари глав -> общее резюме книги (5-7 предложений)."""
        print("   Reduce: генерирую общее резюме книги...")

        all_summaries = "\n".join(
            f"Часть {cs['part']}, Глава {cs['chapter']}: {cs['summary']}"
            for cs in chapter_summaries
        )

        return self._call_llm_summarize(
            all_summaries,
            "На основе кратких содержаний глав составь общее резюме книги в 5-7 предложениях. "
            "Укажи главного героя, основной конфликт, ключевых персонажей и чем заканчивается книга."
        )

    def _extract_aliases(self, chapter_summaries: List[Dict]) -> Dict[str, List[str]]:
        """Извлекает персонажей и их алиасы из chapter_summaries через LLM."""
        print("   Извлекаю персонажей и алиасы...")

        all_summaries = "\n".join(
            f"Часть {cs['part']}, Глава {cs['chapter']}: {cs['summary']}"
            for cs in chapter_summaries
        )

        response = self._call_llm_summarize(
            all_summaries,
            "На основе кратких содержаний глав извлеки всех персонажей книги "
            "и их альтернативные имена (алиасы).\n\n"
            "Верни результат СТРОГО в формате JSON — объект, где ключ — основное имя персонажа, "
            "значение — список его альтернативных имён (прозвища, титулы, фамилии, "
            "обращения вроде «г-жа ...», «г-н ...»).\n\n"
            "Пример формата:\n"
            '{"Иванов": ["Пётр", "Пётр Иванов", "г-н Иванов"]}\n\n'
            "Верни ТОЛЬКО JSON, без пояснений."
        )

        try:
            # Извлекаем JSON из ответа (LLM может обернуть в ```json ... ```)
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("\n", 1)[1]
                json_str = json_str.rsplit("```", 1)[0]
            aliases = json.loads(json_str)
            if isinstance(aliases, dict):
                # Убеждаемся что значения — списки строк
                return {
                    k: [str(a) for a in v] if isinstance(v, list) else []
                    for k, v in aliases.items()
                }
        except (json.JSONDecodeError, ValueError):
            print("   Не удалось распарсить алиасы из LLM, используем пустой словарь")

        return {}

    def _call_llm_summarize(self, text: str, instruction: str) -> str:
        with GigaChat(credentials=self.credentials, verify_ssl_certs=False) as giga:
            response = giga.chat(Chat(
                messages=[
                    Messages(
                        role=MessagesRole.SYSTEM,
                        content=instruction,
                    ),
                    Messages(
                        role=MessagesRole.USER,
                        content=text,
                    ),
                ],
                temperature=0.1,
            ))
        return response.choices[0].message.content.strip()

    @staticmethod
    def _split_text(text: str, chunk_size: int) -> List[str]:
        """Разбивает текст на куски примерно по chunk_size символов по границам абзацев."""
        paragraphs = text.split("\n")
        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for paragraph in paragraphs:
            para_len = len(paragraph) + 1  # +1 for newline
            if current_length + para_len > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(paragraph)
            current_length += para_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks


# =============================================================================
# 3. MetadataExtractor
# =============================================================================

class MetadataExtractor:
    """Извлекает персонажей и локации через spaCy NER + алиасы."""

    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("ru_core_news_sm")
        self.character_aliases: Dict[str, List[str]] = {}
        self.alias_to_canonical: Dict[str, str] = {}

    def set_aliases(self, character_aliases: Dict[str, List[str]]):
        """Устанавливает алиасы персонажей (из кэша или LLM)."""
        self.character_aliases = character_aliases
        self.alias_to_canonical = {}
        for canonical, aliases in character_aliases.items():
            self.alias_to_canonical[canonical.lower()] = canonical
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical

    def extract(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        persons = []
        locations = []
        seen_per = set()
        seen_loc = set()

        for ent in doc.ents:
            if ent.label_ == "PER" and ent.text not in seen_per:
                persons.append(ent.text)
                seen_per.add(ent.text)
            elif ent.label_ == "LOC" and ent.text not in seen_loc:
                locations.append(ent.text)
                seen_loc.add(ent.text)

        return {"persons": persons, "locations": locations}

    def expand_persons_with_aliases(self, persons: List[str]) -> List[str]:
        """Расширяет список персонажей всеми известными алиасами."""
        expanded = set(persons)
        for person in persons:
            canonical = self.alias_to_canonical.get(person.lower())
            if canonical and canonical in self.character_aliases:
                expanded.add(canonical)
                expanded.update(self.character_aliases[canonical])
        return list(expanded)


# =============================================================================
# 4. QueryExpander
# =============================================================================

class QueryExpander:
    """Расширяет запросы алиасами персонажей для BM25 поиска."""

    def __init__(self):
        self.character_aliases: Dict[str, List[str]] = {}

    def set_aliases(self, character_aliases: Dict[str, List[str]]):
        self.character_aliases = character_aliases

    def expand_query(self, query: str) -> str:
        query_lower = query.lower()
        expanded_terms = []

        for canonical, aliases in self.character_aliases.items():
            all_names = [canonical] + aliases
            for name in all_names:
                if name.lower() in query_lower:
                    expanded_terms.extend(
                        a for a in all_names if a.lower() not in query_lower
                    )
                    break

        if expanded_terms:
            return query + " " + " ".join(expanded_terms)
        return query


# =============================================================================
# 5. ChunkProcessor — sentence-aware chunking
# =============================================================================

class ChunkProcessor:
    """Разбивает текст на чанки по предложениям, не пересекая границы глав."""

    def __init__(self, nlp=None, metadata_extractor: Optional[MetadataExtractor] = None):
        self.nlp = nlp or spacy.load("ru_core_news_sm")
        self.metadata_extractor = metadata_extractor or MetadataExtractor(self.nlp)

    def process_sections(
        self,
        sections: List[Dict],
        chunk_size: int = 900,
        sentence_overlap: int = 2,
    ) -> List[Dict]:
        all_chunks = []

        for section in sections:
            part = section["part"]
            chapter = section["chapter"]
            text = section["text"]

            sentences = self._split_sentences(text)
            if not sentences:
                continue

            section_chunks = self._group_sentences(
                sentences, chunk_size, sentence_overlap
            )

            for i, chunk_text in enumerate(section_chunks):
                metadata = self.metadata_extractor.extract(chunk_text)
                all_chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"p{part}_ch{chapter}_c{i}",
                    "part": part,
                    "chapter": chapter,
                    "persons": metadata["persons"],
                    "locations": metadata["locations"],
                })

        return all_chunks

    def _split_sentences(self, text: str) -> List[str]:
        max_length = self.nlp.max_length
        if len(text) > max_length:
            self.nlp.max_length = len(text) + 1000

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        self.nlp.max_length = max_length
        return sentences

    def _group_sentences(
        self,
        sentences: List[str],
        chunk_size: int,
        sentence_overlap: int,
    ) -> List[str]:
        chunks = []
        i = 0

        while i < len(sentences):
            current_sentences = []
            current_length = 0

            j = i
            while j < len(sentences):
                sent = sentences[j]
                new_length = current_length + len(sent) + (1 if current_sentences else 0)
                if current_sentences and new_length > chunk_size:
                    break
                current_sentences.append(sent)
                current_length = new_length
                j += 1

            if current_sentences:
                chunks.append(" ".join(current_sentences))

            added = len(current_sentences)
            step = max(1, added - sentence_overlap)
            i += step

        return chunks


# =============================================================================
# 6. HybridRetriever — retrieve + cross-encoder rerank
# =============================================================================

class HybridRetriever:
    """
    Двухэтапный поиск:
    - Stage 1: семантический (ChromaDB) + BM25 -> объединённый пул кандидатов
    - Stage 2: cross-encoder reranker + metadata boost
    """

    def __init__(
        self,
        metadata_extractor: MetadataExtractor,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        metadata_boost: float = 0.1,
        candidate_pool_size: int = 15,
    ):
        self.metadata_extractor = metadata_extractor
        self.metadata_boost = metadata_boost
        self.candidate_pool_size = candidate_pool_size
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunk_tokens: List[List[str]] = []

        print("   Загружаю cross-encoder reranker...")
        self.reranker = FlagReranker(reranker_model_name, use_fp16=True)

    def build_bm25_index(self, chunks: List[Dict]):
        self.chunk_tokens = [
            chunk["text"].lower().split() for chunk in chunks
        ]
        self.bm25_index = BM25Okapi(self.chunk_tokens)

    def get_bm25_top_candidates(
        self, query: str, all_chunks: List[Dict], top_n: int = 30
    ) -> List[Dict]:
        if not self.bm25_index:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)

        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_n]

        return [all_chunks[i] for i in scored_indices if scores[i] > 0]

    def calculate_metadata_score(
        self, query_metadata: Dict, chunk_metadata: Dict
    ) -> float:
        score = 0.0

        query_persons = set(self.metadata_extractor.expand_persons_with_aliases(
            query_metadata.get("persons", [])
        ))
        chunk_persons = set(self.metadata_extractor.expand_persons_with_aliases(
            chunk_metadata.get("persons", [])
        ))
        if query_persons and query_persons & chunk_persons:
            score += 0.5

        query_locations = set(query_metadata.get("locations", []))
        chunk_locations = set(chunk_metadata.get("locations", []))
        if query_locations and query_locations & chunk_locations:
            score += 0.5

        return min(score, 1.0)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        query_metadata: Dict,
    ) -> List[Dict]:
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        raw_scores = self.reranker.compute_score(pairs)

        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]

        min_s = min(raw_scores)
        max_s = max(raw_scores)
        if max_s > min_s:
            norm_scores = [(s - min_s) / (max_s - min_s) for s in raw_scores]
        else:
            norm_scores = [1.0] * len(raw_scores)

        scored = []
        for i, candidate in enumerate(candidates):
            meta_score = self.calculate_metadata_score(
                query_metadata,
                {"persons": candidate.get("persons", []),
                 "locations": candidate.get("locations", [])},
            )
            final = norm_scores[i] + self.metadata_boost * meta_score
            scored.append({**candidate, "_score": final})

        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored


# =============================================================================
# 7. RAGPipeline — оркестратор
# =============================================================================

class RAGPipeline:
    """Главный оркестратор RAG v5."""

    def __init__(self, book_path: str):
        self.book_path = book_path
        self.chunks: List[Dict] = []
        self.collection = None
        self.book_summary: str = ""

        print("   Загружаю spaCy модель...")
        self.nlp = spacy.load("ru_core_news_sm")

        self.structure_parser = StructureParser()
        self.metadata_extractor = MetadataExtractor(self.nlp)
        self.chunk_processor = ChunkProcessor(self.nlp, self.metadata_extractor)
        self.hybrid_retriever = HybridRetriever(self.metadata_extractor)
        self.query_expander = QueryExpander()
        self.book_summarizer = BookSummarizer()

        print("   Загружаю bge-m3 эмбеддинги...")
        self.embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def load_and_index_book(self, chunk_size: int = 900, sentence_overlap: int = 2):
        print("1. Загружаю текст книги...")
        with open(self.book_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"   Размер: {len(text)} символов")

        print("2. Парсю структуру (части и главы)...")
        sections = self.structure_parser.parse(text)
        print(f"   Найдено {len(sections)} глав")
        for s in sections:
            preview = s["text"][:60].replace("\n", " ")
            print(f"   Часть {s['part']}, Глава {s['chapter']}: {preview}...")

        print("2.5. Генерирую/загружаю суммаризацию (map-reduce)...")
        summaries = self.book_summarizer.get_or_create(self.book_path, sections)
        self.book_summary = summaries["book_summary"]
        chapter_summaries = summaries["chapter_summaries"]
        character_aliases = summaries.get("character_aliases", {})
        print(f"   Общее резюме: {self.book_summary[:100]}...")
        print(f"   Суммари глав: {len(chapter_summaries)} шт.")
        print(f"   Персонажей с алиасами: {len(character_aliases)}")

        # Устанавливаем алиасы в компоненты
        self.metadata_extractor.set_aliases(character_aliases)
        self.query_expander.set_aliases(character_aliases)

        print("3. Разбиваю на чанки (sentence-aware)...")
        self.chunks = self.chunk_processor.process_sections(
            sections, chunk_size, sentence_overlap
        )
        if not self.chunks:
            raise RuntimeError("Не удалось разбить текст на чанки. Проверьте структуру книги.")
        avg_size = sum(len(c["text"]) for c in self.chunks) // len(self.chunks)
        print(f"   Получилось {len(self.chunks)} чанков (средний размер: {avg_size} символов)")

        # Добавляем summary-чанки
        print("3.5. Добавляю summary-чанки в индекс...")
        for cs in chapter_summaries:
            summary_text = (
                f"Краткое содержание Части {cs['part']}, "
                f"Главы {cs['chapter']}: {cs['summary']}"
            )
            metadata = self.metadata_extractor.extract(summary_text)
            self.chunks.append({
                "text": summary_text,
                "chunk_id": f"summary_p{cs['part']}_ch{cs['chapter']}",
                "part": cs["part"],
                "chapter": cs["chapter"],
                "persons": metadata["persons"],
                "locations": metadata["locations"],
            })
        print(f"   Итого чанков (с summary): {len(self.chunks)}")

        print("4. Строю BM25 индекс...")
        self.hybrid_retriever.build_bm25_index(self.chunks)

        print("5. Индексирую в ChromaDB...")
        self._create_vector_store()
        print(f"   В хранилище {self.collection.count()} чанков")

    def _build_system_prompt(self) -> str:
        return (
            "Ты — помощник по книге.\n\n"
            f"ОБЩЕЕ СОДЕРЖАНИЕ КНИГИ:\n{self.book_summary}\n\n"
            "ПРАВИЛА:\n"
            "1. Отвечай на основе предоставленных фрагментов текста. "
            "Общее содержание книги выше можно использовать как ДОПОЛНИТЕЛЬНЫЙ контекст "
            "для понимания сюжета, но приоритет — у фрагментов.\n"
            "2. Если информации в фрагментах НЕДОСТАТОЧНО для полного ответа, "
            "явно укажи, какой именно информации не хватает.\n"
            "3. НЕ додумывай, НЕ экстраполируй, НЕ добавляй деталей, "
            "которых нет в предоставленных фрагментах и общем содержании.\n"
            "4. Если фрагменты не содержат информации по вопросу, "
            "но общее содержание содержит — ответь на основе общего содержания, "
            "указав, что ответ основан на общем содержании книги.\n"
            "5. При цитировании указывай, из какой части и главы "
            "взята информация.\n"
            "Отвечай на русском языке."
        )

    def _create_vector_store(self):
        client = chromadb.Client()

        try:
            client.delete_collection("book_chunks_v5")
        except Exception:
            pass

        self.collection = client.create_collection(
            name="book_chunks_v5",
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for start in range(0, len(self.chunks), batch_size):
            end = min(start + batch_size, len(self.chunks))
            batch = self.chunks[start:end]

            documents = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [
                {
                    "part": c["part"],
                    "chapter": c["chapter"],
                    "persons": ",".join(c["persons"]),
                    "locations": ",".join(c["locations"]),
                }
                for c in batch
            ]

            result = self.embed_model.encode(documents, batch_size=12, max_length=1024)
            embeddings = result["dense_vecs"].tolist()

            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            print(f"   Обработано {end}/{len(self.chunks)} чанков...")

    def retrieve(self, query: str, top_k: int = 12) -> List[Dict]:
        if not self.collection:
            raise RuntimeError("Коллекция не инициализирована. Вызовите load_and_index_book()")

        pool_size = self.hybrid_retriever.candidate_pool_size

        # 1. Семантический поиск
        query_result = self.embed_model.encode([query], max_length=512)
        query_embedding = query_result["dense_vecs"].tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(pool_size, self.collection.count()),
        )

        semantic_candidates = {}
        for doc, meta, cid in zip(
            results["documents"][0], results["metadatas"][0], results["ids"][0]
        ):
            persons = [p.strip() for p in meta.get("persons", "").split(",") if p.strip()]
            locations = [l.strip() for l in meta.get("locations", "").split(",") if l.strip()]
            semantic_candidates[cid] = {
                "text": doc,
                "chunk_id": cid,
                "part": meta.get("part", 0),
                "chapter": meta.get("chapter", 0),
                "persons": persons,
                "locations": locations,
            }

        # 2. BM25 поиск (с расширением алиасами)
        expanded_query = self.query_expander.expand_query(query)
        bm25_candidates = self.hybrid_retriever.get_bm25_top_candidates(
            expanded_query, self.chunks, top_n=pool_size
        )

        # 3. Merge + dedup
        merged = dict(semantic_candidates)
        for c in bm25_candidates:
            if c["chunk_id"] not in merged:
                merged[c["chunk_id"]] = c

        candidates = list(merged.values())

        # 4. Cross-encoder reranking
        query_metadata = self.metadata_extractor.extract(query)
        reranked = self.hybrid_retriever.rerank(query, candidates, query_metadata)

        return reranked[:top_k]

    def decompose_query(self, query: str) -> List[str]:
        """Просит LLM решить: разбить запрос на подвопросы или вернуть как есть."""
        credentials = os.environ.get("GIGACHAT_CREDENTIALS")
        if not credentials:
            return [query]

        try:
            with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
                response = giga.chat(Chat(
                    messages=[
                        Messages(
                            role=MessagesRole.SYSTEM,
                            content=(
                                "Ты анализируешь вопросы пользователя о книге и решаешь, "
                                "нужно ли разбить вопрос на подвопросы для поиска по тексту.\n\n"
                                "Вопрос НУЖНО разбить, если он:\n"
                                "- Требует информации из разных частей книги (временная динамика, эволюция, сравнение)\n"
                                "- Просит перечислить несколько сущностей (всех персонажей, все события)\n"
                                "- Затрагивает несколько персонажей или тем одновременно\n\n"
                                "Вопрос НЕ нужно разбивать, если он:\n"
                                "- Про конкретную сцену, событие или факт\n"
                                "- Достаточно простой для одного поиска\n\n"
                                "ФОРМАТ ОТВЕТА:\n"
                                "- Если разбиваешь: верни 2-4 подвопроса, каждый на новой строке. "
                                "Каждый подвопрос должен быть направлен на поиск конкретных сцен и событий.\n"
                                "- Если НЕ разбиваешь: верни исходный вопрос без изменений.\n"
                                "- Без нумерации, маркеров и пояснений.\n"
                            ),
                        ),
                        Messages(
                            role=MessagesRole.USER,
                            content=query,
                        ),
                    ],
                    temperature=0.0,
                ))

            sub_queries = [
                line.strip().lstrip("0123456789.-) ")
                for line in response.choices[0].message.content.split("\n")
                if line.strip()
            ]
            return sub_queries if sub_queries else [query]
        except Exception:
            return [query]

    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        credentials = os.environ.get("GIGACHAT_CREDENTIALS")
        if not credentials:
            raise RuntimeError("Установите переменную GIGACHAT_CREDENTIALS")

        context = self._format_context(chunks)

        with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
            response = giga.chat(Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=self._build_system_prompt()),
                    Messages(
                        role=MessagesRole.USER,
                        content=f"Контекст из книги:\n\n{context}\n\nВопрос: {query}",
                    ),
                ],
                temperature=0.1,
            ))

        return response.choices[0].message.content

    def _format_context(self, chunks: List[Dict]) -> str:
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Фрагмент {i} | Часть {chunk['part']}, Глава {chunk['chapter']}]"
            formatted.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(formatted)

    def process_question(self, query: str) -> str:
        print("   Анализирую запрос...")
        sub_queries = self.decompose_query(query)

        if len(sub_queries) > 1:
            print(f"   Разбит на подвопросы: {sub_queries}")
            all_chunks: List[Dict] = []
            seen_ids = set()
            for sq in sub_queries:
                sq_chunks = self.retrieve(sq, top_k=12)
                for c in sq_chunks:
                    if c["chunk_id"] not in seen_ids:
                        all_chunks.append(c)
                        seen_ids.add(c["chunk_id"])
            relevant_chunks = all_chunks[:20]
        else:
            print("   Простой запрос — ищу напрямую")
            relevant_chunks = self.retrieve(query, top_k=12)

        print(f"   Найдено {len(relevant_chunks)} релевантных фрагментов:")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"   [{i}] Ч.{chunk['part']} Гл.{chunk['chapter']}:\n{chunk['text']}\n")

        print("   Генерирую ответ...")
        return self.generate_answer(query, relevant_chunks)


# =============================================================================
# CLI
# =============================================================================

def main():
    import sys

    book_path = os.path.join(os.path.dirname(__file__), "book.txt")
    if not os.path.exists(book_path):
        print(f"Ошибка: файл {book_path} не найден")
        sys.exit(1)

    print("=" * 70)
    print("RAG Pipeline v5 — Map-Reduce суммаризация + bge-m3 + reranker")
    print("=" * 70)

    pipeline = RAGPipeline(book_path)
    pipeline.load_and_index_book()

    print("\n" + "=" * 70)
    print("Система готова! Задавайте вопросы по книге.")
    print("Для выхода введите 'выход' или 'exit'.")
    print("=" * 70)

    while True:
        print()
        query = input("Ваш вопрос: ").strip()

        if not query:
            continue

        if query.lower() in ("выход", "exit", "quit"):
            print("До свидания!")
            break

        print()
        try:
            answer = pipeline.process_question(query)
            print(f"\nОтвет: {answer}")
        except Exception as e:
            print(f"\nОшибка: {e}")


if __name__ == "__main__":
    main()
