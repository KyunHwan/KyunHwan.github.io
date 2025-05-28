---
layout: default
title: Poems
class: poems-page
permalink: /poems/
---

# Poems

<div class="card-grid">
  {% for poem in site.poems %}
    <a class="card" href="{{ poem.url | relative_url }}">
      <h2>{{ poem.title }}</h2>
      <p>{{ poem.excerpt | strip_html | truncate: 100 }}</p>
    </a>
  {% endfor %}
</div>