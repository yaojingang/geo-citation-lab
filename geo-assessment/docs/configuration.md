# 配置说明

应用通过进程环境变量读取配置。相对路径以项目根目录为基准。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `APP_ENV` | `production` | 运行环境名称 |
| `APP_DEBUG` | `0` | 本地诊断时可设为 `1`，生产保持 `0` |
| `APP_BASE_PATH` | 空 | 安装在子目录时填写路径，例如 `/geo` |
| `GEO_PUBLIC_URL` | 空 | 证书二维码使用的公开 HTTPS 根地址，例如 `https://ai.laoyao.cn/geo`；生产环境必须配置，留空只支持本机回环开发地址 |
| `APP_TIMEZONE` | `Asia/Shanghai` | 页面日期显示时区 |
| `GEO_DATA_DIR` | `storage` | 数据库、密钥、日志和备份的持久化根目录；版本化部署必须指向版本目录外的共享位置 |
| `GEO_LEGACY_DATA_DIR` | 当前版本的 `storage` | 供 `deploy/install.sh` 检测旧数据；共享目录为空时会停止安装并提示迁移，避免初始化新数据库 |
| `GEO_DB_PATH` | `$GEO_DATA_DIR/app.sqlite` | SQLite 数据库路径；显式配置时覆盖数据根目录默认值 |
| `GEO_LOG_DIR` | `$GEO_DATA_DIR/logs` | JSONL 错误日志目录；显式配置时覆盖数据根目录默认值 |
| `GEO_BACKUP_DIR` | `$GEO_DATA_DIR/backups` | 最近 7 份一致性备份目录；显式配置时覆盖数据根目录默认值 |
| `GEO_COOKIE_SECURE` | `auto` | `1` 强制 Secure，`0` 关闭，`auto` 跟随 HTTPS |
| `GEO_TRUST_PROXY` | `0` | 只在受信反向代理固定覆盖来源头时设为 `1` |
| `GEO_APP_KEY` | 空 | 可由环境注入；空值时读取 `$GEO_DATA_DIR/app.key` |
| `GEO_BAIDU_ANALYTICS_ID` | 空 | 可选的 32 位百度统计 ID；留空时不输出统计脚本，也不在 CSP 中放行统计域名 |

首次运行 `php bin/console app:install` 会创建存储目录、生成 64 字符密钥、迁移数据库并导入 `geo-30-v1.2`。安装过程幂等，现有密钥与同指纹题集会保留。已有 `geo-30-v1.1` 作答记录继续使用当时保存的题目、维度名称和题集版本。升级现有环境时，先运行 `php bin/console db:migrate`，再运行 `php bin/console questions:import database/seeds/geo-30-v1.2.json`。

单目录本地运行可以使用默认 `storage/`。生产和版本化发布应把 `GEO_DATA_DIR` 设为独立共享目录，例如 `/srv/geo-assessment/shared`。应用用户需要读写该目录，Web 文档根只指向 `public/`。`app.key` 使用 `0600`，数据库、日志与备份禁止 Web 直接访问。

启用百度统计时，未登录首页会输出初始化代码，并为这段代码动态生成精确的 CSP 哈希。身份、答题、报告、证书及错误页面不会加载第三方统计。无效 ID 会被关闭，`app:health` 会报告配置错误。部署者需要在面向用户的隐私说明中说明第三方统计的数据范围和保留政策。
