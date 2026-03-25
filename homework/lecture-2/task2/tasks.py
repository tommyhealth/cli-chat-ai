from gigachat_client import GigaChatClient
from models import Resume

def parse_resume_task(client: GigaChatClient, resume_text: str) -> Resume:
    """Спарси резюме и верни структурированный объект Resume"""
    response = client.parse_resume(resume_text)
    resume = Resume(**response)
    return resume

def format_resume_output(resume: Resume) -> str:
    """Форматируй Resume для красивого вывода"""
    output = []
    output.append("=" * 70)
    output.append(f"ФИО: {resume.name}")

    if resume.email:
        output.append(f"EMAIL: {resume.email}")
    if resume.phone:
        output.append(f"ТЕЛЕФОН: {resume.phone}")

    output.append("\nНАВЫКИ:")
    if resume.skills:
        for skill in resume.skills:
            output.append(f"  • {skill}")
    else:
        output.append("  (нет данных)")

    output.append("\nОПЫТ РАБОТЫ:")
    if resume.experience:
        for exp in resume.experience:
            output.append(f"  {exp.position} в {exp.company} ({exp.years} лет)")
            if exp.description:
                output.append(f"    {exp.description}")
    else:
        output.append("  (нет данных)")

    output.append("\nОБРАЗОВАНИЕ:")
    if resume.education:
        for edu in resume.education:
            output.append(f"  {edu.degree} от {edu.university} ({edu.graduation_year})")
    else:
        output.append("  (нет данных)")

    output.append("=" * 70)

    return "\n".join(output)
