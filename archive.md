---
layout: page
title: 归档
permalink: /archive/
---

<!-- 按时间线列出所有文章 -->
{% assign current_year = "" %}
{% for post in site.posts %}
{% assign post_year = post.date | date: "%Y" %}
{% if post_year != current_year %}
{% unless forloop.first %}
</ul>
{% endunless %}
<h3>{{ post_year }}</h3>
<ul>
{% assign current_year = post_year %}
{% endif %}
<li><span>{{ post.date | date: "%m-%d" }}</span> <a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
{% if site.posts.size > 0 %}
</ul>
{% endif %}
