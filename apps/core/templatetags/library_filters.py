from django import template

register = template.Library()


@register.filter
def in_titles(lesson_title, comma_separated_titles):
    """Usage: {{ lesson.title|in_titles:"Halo,Selamat Pagi,Selamat Siang" }}"""
    wanted = [t.strip() for t in comma_separated_titles.split(",")]
    return lesson_title in wanted