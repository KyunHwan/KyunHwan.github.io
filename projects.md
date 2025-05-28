---
layout: default
title: Projects
class: projects-page
permalink: /projects/
---

# Projects

<div class="card-grid">
  {% for project in site.projects %}
    <a class="card" href="{{ project.url | relative_url }}">
      <h2>{{ project.title }}</h2>
      <p>{{ project.excerpt | strip_html | truncate: 100 }}</p>
    </a>
  {% endfor %}
</div>