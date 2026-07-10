---
layout: post
title: "如何写一篇新文章"
date: 2026-07-08 10:00:00 +0800
categories: 指南
tags: [教程, jekyll, markdown]
---

想发新文章,照着做:

## 1. 在 `_posts/` 目录下新建文件

文件名必须是这样:

```
年-月-日-英文标题.md
```

比如 `2026-07-08-my-first-post.md`。

## 2. 文件开头写这段（叫 front matter）

```yaml
---
layout: post
title: "文章标题"
date: 2026-07-08 10:00:00 +0800
categories: 指南
tags: [教程, jekyll, markdown]
---
```

## 3. 下面用 Markdown 写正文

- **加粗**、*斜体*
- 列表
- `代码`
- [链接](https://example.com)

```python
print("代码块")
```

> 引用块长这样

## 4. 保存后上传

```bash
git add .
git commit -m "新文章"
git push
```

等 1-2 分钟刷新网站就能看到了。

---

*这是占位示例文章，你写好第一篇真实的之后，把这个文件删掉就行。*
