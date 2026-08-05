# 参与贡献

欢迎提交程序缺陷、题目修订、部署兼容性和报告可视化方面的改进。开始修改前，请先在 Issues 中确认问题范围，较大的功能建议先说明使用场景和数据边界。

## 开发环境

运行时支持 PHP 7.3.5+ 与 SQLite 3.24+。开发测试需要 PHP 8.2+、Composer、Node.js 和 Docker。

```bash
composer install
php bin/console app:install
composer validate --strict
composer verify
```

`composer verify` 会执行 PHP 测试、JavaScript 测试、题库校验、健康检查和 PHP 7.3 Docker 兼容验证。

## 提交范围

- 不要提交 `storage/app.sqlite`、`storage/app.key`、日志、备份、真实姓名或作答导出
- 数据库结构变化需要新增迁移文件，已有迁移保持不可变
- 题目修订需要同步答案、解析、原理、来源和内容指纹验证
- 新的第三方代码或内容需要更新 `THIRD_PARTY_NOTICES.md` 和许可说明
- UI 修改需要检查桌面端、移动端、键盘操作、200% 缩放和打印报告

题目来源规则见 [`docs/question-provenance.md`](docs/question-provenance.md)，发布检查见 [`docs/release-checklist.md`](docs/release-checklist.md)。

## Pull Request

PR 描述请写明改动目的、验证命令、数据库影响、隐私影响和截图。请把功能改动与无关格式化拆开，方便审查和回滚。
