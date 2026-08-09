# GEO Assessment

基于 PHP 7.3.5+ 与 SQLite 3.24+ 的 GEO 在线测试。系统提供 30 道初学者友好的场景题、30 分钟倒计时、每个浏览器身份最多 10 次机会、永久登录态、历史记录与多图表详细报告。题目以国内应用为主，研究证据来自仓库内 54 篇 GEO、AI 搜索与 RAG 研究，以及海外平台实验和 214,119 条国内引用记录。

## 本地运行

生产运行时兼容 PHP 7.3.5+。PHP 7.3 已停止安全维护，新的公开部署建议使用仍在安全维护期内的 PHP 版本。完整开发测试使用 PHPUnit 11，需要 PHP 8.2+；旧版 PHP 服务器可使用包含生产依赖的 Release 压缩包，无需在服务器执行 Composer。

```bash
composer install
php bin/console app:install
php -S 127.0.0.1:8080 -t public public/router.php
```

打开 `http://127.0.0.1:8080`。应用数据写入 `storage/app.sqlite`，应用密钥写入 `storage/app.key`。

## 验证

```bash
composer validate --strict
composer lint
composer test
php bin/console questions:verify
php bin/console app:health
bash tests/Smoke/http.sh http://127.0.0.1:8080
```

## 主要能力

- 姓名规范化、浏览器匿名令牌、10 年滚动会话与带确认的身份切换
- 30 题和选项随机顺序、中性展示序号、服务端截止时间、断点续答与原子保存
- 20 道单选、10 道多选，完整集合匹配，总分固定 100 分
- 国内 18 题、通用 9 题和海外 3 题，题干无需论文或数据仓库背景
- 六维诊断、难度表现、逐题用时、题目矩阵、历次趋势和群体分位
- 30 道题的本人选择、正确答案、选项理由、解析、原理和证据来源
- 纯白响应式 UI、键盘答题、无 JavaScript 表单流程、A4 打印报告
- CSRF、CSP、禁止个人页面缓存、参数化 SQL、报告归属校验、限流与结构化错误日志
- 一致性备份、SHA-256 校验、题库验证与健康检查

## 运维文档

- [配置说明](docs/configuration.md)
- [部署说明](docs/deployment.md)
- [备份与恢复](docs/backup-and-restore.md)
- [发布检查清单](docs/release-checklist.md)
- [题目来源与编写边界](docs/question-provenance.md)
- [题库 v1.2 逐题审查清单](docs/question-review-v1.2.md)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [变更记录](CHANGELOG.md)

产品规范见 [DESIGN.md](DESIGN.md)，第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 在线体验与开源计划

- 在线体验：<https://ai.laoyao.cn/geo/>
- GitHub 开源计划：[docs/github-open-source-plan.md](docs/github-open-source-plan.md)

公共源码默认关闭访问统计。部署者可以通过 `GEO_BAIDU_ANALYTICS_ID` 配置自己的百度统计 ID；启用后只在未登录首页加载，并需要在站点隐私说明中披露统计用途。身份、答题、报告与证书页面不会加载第三方统计。官方演示站点的参数位于 `deploy/examples/ai.laoyao.cn/`，通用部署从 `deploy/install.sh` 与 `deploy/nginx-subdirectory.conf.example` 开始。

## 许可范围

- PHP、JavaScript、CSS、安装脚本和测试代码采用 [MIT License](LICENSE-CODE)
- 题目、解析、产品文档与原创可视化内容采用 [CC BY 4.0](LICENSE-CONTENT)
- Chart.js 及其他第三方材料维持各自许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- 本地 SQLite 数据库、应用密钥、日志、备份、开发依赖和发布压缩包不进入 Git 仓库
