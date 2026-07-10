---
layout: page
title: 标签
permalink: /tags/
---

<!-- 自动列出所有标签及对应文章 -->
{% assign rawtags = "" %}
{% for post in site.posts %}
  {% assign ttags = post.tags | join: '|' | append: '|' %}
  {% assign rawtags = rawtags | append: ttags %}
{% endfor %}
{% assign rawtags = rawtags | split: '|' | uniq %}

{% for tag in rawtags %}
  <h3 id="{{ tag | slugify }}">{{ tag }}</h3>
  <ul>
  {% for post in site.posts %}
    {% if post.tags contains tag %}
      <li><a href="{{ post.url }}">{{ post.title }}</a> <small>({{ post.date | date: "%Y-%m-%d" }})</small></li>
    {% endif %}
  {% endfor %}
  </ul>
{% endfor %}
