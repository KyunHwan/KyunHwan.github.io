---
layout: default
title: Poems
---

# My Poems

A collection of my poetry:

{% for poem in site.poems %}
## [{{ poem.title }}]({{ poem.url | relative_url }})
{{ poem.excerpt }}
{% endfor %} 