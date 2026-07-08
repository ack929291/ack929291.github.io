# 我的博客

基于 Jekyll + GitHub Pages，使用 minima 主题。

## 本地预览（可选）

需要 Ruby 环境，一般不用，直接 push 到 GitHub 让它编译就行：

```bash
bundle install        # 第一次运行
bundle exec jekyll serve   # 启动后访问 http://localhost:4000
```

## 发布新文章

1. 在 `_posts/` 下新建文件，命名格式：`YYYY-MM-DD-标题.md`
2. 文件开头写好 front matter（layout/title/date/categories）
3. `git add . && git commit -m "新文章" && git push`
4. 等 1-2 分钟，刷新网站

## 相关文件说明

- `_config.yml` — 站点配置（标题、作者、主题等）
- `_posts/` — 文章目录
- `index.md` — 首页
- `about.md` — 关于页
