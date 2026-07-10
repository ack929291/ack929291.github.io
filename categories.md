---
layout: page
title: 分类
permalink: /categories/
---

<!-- 自动列出所有分类及对应文章 -->
{% assign rawcats = "" %}
{% for post in site.posts %}
  {% assign tcats = post.categories | join: '|' | append: '|' %}
  {% assign rawcats = rawcats | append: tcats %}
{% endfor %}
{% assign rawcats = rawcats | split: '|' | uniq %}

{% for cat in rawcats %}
  <h3 id="{{ cat | slugify }}">{{ cat }}</h3>
  <ul>
  {% for post in site.posts %}
    {% if post.categories contains cat %}
      <li><a href="{{ post.url }}">{{ post.title }}</a> <small>({{ post.date | date: "%Y-%m-%d" }})</small></li>
    {% endif %}
  {% endfor %}
  </ul>
{% endfor %}
