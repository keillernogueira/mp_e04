from django import template                                                                                                                                                        
from django.template.defaultfilters import stringfilter

register = template.Library()

@register.filter
@stringfilter
def dash2space(s):
    return s.replace('_', ' ') 