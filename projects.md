---
layout: default
title: Projects
---

# My Projects

Here are some of the projects I've worked on:

{% for project in site.projects %}
## [{{ project.title }}]({{ project.url | relative_url }})
{{ project.excerpt }}
{% endfor %} 