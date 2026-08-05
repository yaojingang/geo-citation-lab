# 发布检查清单

## 构建门槛

- [ ] `composer validate --strict` 通过
- [ ] `composer lint` 通过
- [ ] `composer test` 通过
- [ ] `composer test:js` 通过
- [ ] `composer test:php73` 的 PHP 7.3 全量语法检查通过
- [ ] `composer test:php73-runtime` 的 PHP 7.3 安装、迁移、题库和健康检查通过
- [ ] `php bin/console questions:verify` 通过
- [ ] `php bin/console app:health` 通过
- [ ] Chart.js 固定为 4.5.1，SHA-256 与许可清单一致
- [ ] 部署包不含论文 PDF、用户数据库、密钥、日志、备份和开发依赖
- [ ] `bash tools/build-release.sh assessment-v0.1.0` 生成 ZIP 与 SHA-256
- [ ] `bash tools/verify-release.sh dist/geo-assessment-0.1.0.zip` 完成归档与隔离安装验证

## 功能验收

- [ ] 姓名创建身份后直接进入首题，刷新与重启浏览器后身份仍有效
- [ ] 单选、多选、空答案、上一题、下一题、题号跳转和断点继续正确
- [ ] 截止时间由服务端结算，重复交卷得到同一报告
- [ ] 第 11 次测试被阻止，同名测试者保持隔离
- [ ] 首页显示历史成绩，旧报告使用当次题目快照
- [ ] 报告包含 8 类视图、30 题详情、正确答案、解析、原理和来源
- [ ] 切换身份需二次确认并撤销当前会话，确认姓名后可级联删除本人记录

## 浏览器与可用性

- [ ] 1440×900、1280×720、768×1024、390×844、375×667、320×568 无横向溢出
- [ ] 典型答题在 1280×720 与 375×667 同屏展示题干、选项和操作
- [ ] 320×568 长选项仅在选择区滚动，操作区保持可见
- [ ] 键盘、焦点、200% 缩放、减少动画和无 JavaScript 流程可用
- [ ] A4 打印展示全部题目详情，打印姓名开关生效

## 安全与运维

- [ ] CSRF、CSP、`Cache-Control: no-store`、`X-Content-Type-Options`、`Referrer-Policy` 与 `frame-ancestors` 生效
- [ ] `GEO_TRUST_PROXY=0` 时伪造的 `X-Forwarded-Proto` 无法影响 Cookie 安全属性
- [ ] 修改报告或测试 UUID 无法访问他人数据
- [ ] 日志不包含姓名、答案、Cookie、IP、CSRF、SQL 参数和堆栈
- [ ] Web 文档根只指向 `public/`
- [ ] 发布前备份已创建并验证，恢复步骤已明确
- [ ] `bash tests/Smoke/http.sh https://target.example` 通过
- [ ] 公共源码默认不加载第三方统计；演示站点启用统计时首页已披露用途
