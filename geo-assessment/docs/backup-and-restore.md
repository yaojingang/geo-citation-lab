# 备份与恢复

## 创建与验证

```bash
php bin/console backup:create
php bin/console backup:verify
```

SQLite 3.27+ 使用 `VACUUM INTO` 生成一致性副本，SQLite 3.24–3.26 在写锁保护下复制数据库并恢复 WAL 模式。随后执行 SHA-256、`integrity_check`、9 张业务表、唯一活动题集、30 题与 100 分约束检查。系统保留最近 7 份 `.sqlite` 及对应 `.sha256`。

生产建议每天调用一次 `backup:create`，并把合格副本同步到具有独立故障域的受控存储。备份包含姓名和完整作答，访问权限需要等同于生产数据库。

## 恢复流程

恢复会覆盖现有数据，需要安排停写窗口：

1. 停止 PHP 写流量。
2. 复制当前数据库和 `-wal`、`-shm` 文件到隔离目录。
3. 对目标备份运行 `backup:verify /absolute/path/backup.sqlite`。
4. 将合格备份复制为配置中的 `GEO_DB_PATH`，恢复应用用户读写权限。
5. 运行 `php bin/console app:health`。
6. 使用回环地址读取首页和一份已有报告。
7. 恢复流量并观察错误日志。

数据库结构采用向前迁移。结构问题通过新的迁移修复；数据恢复使用迁移前的合格备份。
