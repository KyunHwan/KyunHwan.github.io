---
layout: default
title: Projects
class: projects-page
permalink: /projects/
---

# Projects
{: .page-title}

<div class="card-grid">
  {% for project in site.projects %}
    <a class="card" href="{{ project.url | relative_url }}">
      <h2>{{ project.title }}</h2>
    </a>
  {% endfor %}
</div>