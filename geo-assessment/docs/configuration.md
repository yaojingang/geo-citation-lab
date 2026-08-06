# 配置说明

应用通过进程环境变量读取配置。相对路径以项目根目录为基准。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `APP_ENV` | `production` | 运行环境名称 |
| `APP_DEBUG` | `0` | 本地诊断时可设为 `1`，生产保持 `0` |
| `APP_BASE_PATH` | 空 | 安装在子目录时填写路径，例如 `/geo` |
| `APP_TIMEZONE` | `Asia/Shanghai` | 页面日期显示时区 |
| `GEO_DB_PATH` | `storage/app.sqlite` | SQLite 数据库路径 |
| `GEO_LOG_DIR` | `storage/logs` | JSONL 错误日志目录 |
| `GEO_BACKUP_DIR` | `storage/backups` | 最近 7 份一致性备份目录 |
| `GEO_COOKIE_SECURE` | `auto` | `1` 强制 Secure，`0` 关闭，`auto` 跟随 HTTPS |
| `GEO_TRUST_PROXY` | `0` | 只在受信反向代理固定覆盖来源头时设为 `1` |
| `GEO_APP_KEY` | 空 | 可由环境注入；空值时读取 `storage/app.key` |
| `GEO_BAIDU_ANALYTICS_ID` | 空 | 可选的 32 位百度统计 ID；留空时不输出统计脚本，也不在 CSP 中放行统计域名 |

首次运行 `php bin/console app:install` 会创建存储目录、生成 64 字符密钥、迁移数据库并导入 `geo-30-v1.2`。安装过程幂等，现有密钥与同指纹题集会保留。已有 `geo-30-v1.1` 作答记录继续使用当时保存的题目、维度名称和题集版本。升级现有环境时，先运行 `php bin/console db:migrate`，再运行 `php bin/console questions:import database/seeds/geo-30-v1.2.json`。

生产权限建议：应用用户可读源代码，可读写 `storage/`，Web 文档根只指向 `public/`。`storage/app.key` 使用 `0600`，数据库、日志与备份禁止 Web 直接访问。

启用百度统计时，模板会直接输出完整初始化代码，并为这段代码动态生成精确的 CSP 哈希。无效 ID 会被关闭，`app:health` 会报告配置错误。部署者需要在面向用户的隐私说明中说明第三方统计的数据范围和保留政策。
