# GEO Assessment GitHub 开源计划

版本：v1 方案稿

适用目录：`geo-assessment/`

推荐首个公开版本：`assessment-v0.1.0`

## 一、结论

GEO Assessment 先作为 GEO Citation Lab 单仓库中的独立子项目开放源码。这个方案能保留题目与论文、跨平台实验、CN-GEO 数据之间的证据链，同时复用仓库现有的分范围许可、GitHub Issues 和研究品牌。首个版本稳定后，再依据维护节奏和贡献者结构决定是否拆分为独立仓库。

这份计划假设 GEO Citation Lab 会持续作为公开研究仓库维护。如果测试系统形成独立版本节奏、独立维护团队或大量非研究类需求，则在 `assessment-v0.2.0` 后启动仓库拆分评估。

## 二、开源目标与成功标准

### 目标

1. 任何具备 PHP 7.3.5、PDO SQLite 3.24 和 Composer 的开发者，都能在十分钟内完成本地安装
2. 公开源码中不包含姓名、作答记录、SQLite 生产数据、应用密钥、服务器日志或备份
3. 30 道题、100 分权重、题目解析和六维报告具备清晰的来源与许可说明
4. Pull Request 自动完成 PHP、JavaScript、题库、PHP 7.3 兼容性和 HTTP 冒烟验证
5. GitHub Release 提供生产依赖已安装的 ZIP 与 SHA-256，服务器无需运行 Composer
6. README 同时提供在线体验、本地运行、生产部署、数据边界和贡献入口

### 首版验收口径

- `composer validate --strict && composer verify` 全部通过
- PHP 7.3 Docker 兼容测试通过
- 新环境从源码安装后可完成姓名创建、答题、交卷、报告和删除个人记录
- 发布压缩包不含 `storage/app.sqlite`、`storage/app.key`、日志、备份、`vendor` 开发依赖和本地缓存
- 许可证、题库来源、Chart.js 许可和隐私边界均可从 README 到达
- GitHub Actions 中没有未固定主版本的第三方 Action

## 三、范围

### 首次开源包含

- PHP 与 SQLite 应用源码
- 30 道题的结构化题库、权重、答案、解析、原理和来源引用
- 首页、答题页、报告页及响应式样式
- 数据库迁移、安装命令、健康检查、备份和恢复工具
- 单元测试、集成测试、JavaScript 测试、PHP 7.3 兼容测试和 HTTP 冒烟测试
- 通用 Nginx 配置示例和版本化发布脚本
- 贡献、问题反馈、安全披露和许可说明

### 首次开源范围外

- 线上用户姓名、答案、分数、会话、访问日志和统计明细
- 服务器 SSH 信息、真实目录备份、证书、密钥和面板配置
- 论文 PDF 的再分发
- 多租户后台、管理员面板、账号密码体系和云端同步
- SaaS 计费、商业授权和托管服务承诺
- GitHub Pages 运行版本，Pages 无法执行 PHP

## 四、仓库结构

首版保持以下边界：

```text
geo-citation-lab/
├── README.md
├── LICENSE
├── LICENSE-CODE
├── LICENSE-CONTENT
├── THIRD_PARTY_NOTICES.md
└── geo-assessment/
    ├── README.md
    ├── DESIGN.md
    ├── composer.json
    ├── bin/
    ├── config/
    ├── database/
    ├── public/
    ├── src/
    ├── templates/
    ├── tests/
    ├── tools/
    ├── deploy/
    ├── docs/
    └── storage/.gitignore
```

根仓库继续承担研究资料与总许可，`geo-assessment/` 具备独立安装、测试和发布能力。发布归档只从该目录的明确允许列表生成。

## 五、关键决策

### 1. 单仓库子项目优先

题目的证据基础位于同一仓库，单仓库可以让贡献者直接核对论文分类、实验报告和 CN-GEO 说明。独立仓库方案会增加版本同步、来源链接和许可维护成本，首版暂不采用。

### 2. 使用独立标签命名空间

仓库现有 Viewer 使用 `v*` 标签。测试系统使用 `assessment-v*`，首个标签为 `assessment-v0.1.0`，避免触发现有 Viewer 发布工作流，也避免两个产品共享版本号。

### 3. 代码与题目内容分范围许可

- 软件代码：MIT
- 原创题目、解析、报告文案与设计文档：CC BY 4.0
- Chart.js：MIT，并保留本地许可文件
- 论文与第三方数据：仅保留来源引用和研究说明，不把论文 PDF 放入测试系统发布包

新增 `docs/question-provenance.md`，逐类说明题目如何由论文、实验和数据结论转化，并声明题目为原创编写的研究型测验内容。

### 4. 统计代码改为显式配置

公共源码默认关闭百度统计。部署者通过 `GEO_BAIDU_ANALYTICS_ID` 启用自己的统计 ID。README 说明启用统计后的隐私披露责任，测试和本地开发保持无第三方请求。

### 5. 保持 PHP 与 SQLite 的低门槛架构

首版继续使用 PHP 7.3.5+ 和 SQLite 3.24+，不新增 Node 构建、Redis、MySQL 或外部 API。开发测试使用 PHP 8.2+，PHP 7.3 通过 Docker 执行兼容门禁。

## 六、实施阶段

### 阶段一：仓库可公开基线

目标：源码进入 Git，许可、隐私和服务器边界清晰，当前功能保持可运行。

文件工作：

1. 根 `README.md` 增加在线测试与 `geo-assessment/` 入口
2. `geo-assessment/README.md` 增加在线体验、许可范围和开源计划入口
3. 保留 `geo-assessment/.gitignore` 与 `storage/.gitignore`，增加提交内容扫描
4. 新增 `docs/question-provenance.md`，建立题目来源与内容许可说明
5. 把硬编码百度统计 ID 改为可选环境变量，默认值为空
6. 把 `ai.laoyao.cn` 专用部署文件移到 `deploy/examples/ai.laoyao.cn/`
7. 新增通用 `deploy/nginx-subdirectory.conf.example` 和通用安装说明
8. 检查所有文档，不公开服务器凭据、真实数据库、应用密钥和个人记录

阶段验收：

```bash
cd geo-assessment
composer validate --strict
composer verify
git status --ignored --short storage vendor dist .phpunit.cache
```

阶段一可以独立合并。完成后，源码具备公开阅读、克隆和手动安装条件。

### 阶段二：持续集成与可复现发布

目标：每个 Pull Request 自动验证，维护者可以从标签生成固定发布资产。

文件工作：

1. 新增 `.github/workflows/geo-assessment-ci.yml`
2. 使用路径过滤，仅在 `geo-assessment/**` 或工作流自身变化时运行
3. PHP 8.2 与 8.4 执行 Composer 校验、PHPUnit、题库校验和健康检查
4. Docker `php:7.3-cli` 执行全量语法检查和安装运行验证
5. Node 当前 LTS 执行 `node --test tests/Js/*.test.mjs`
6. 新增 `tools/build-release.sh`，从允许列表生成 `geo-assessment-<version>.zip`
7. 新增 `tools/verify-release.sh`，检查归档内容、可执行位、敏感文件和 SHA-256
8. 新增 `.github/workflows/geo-assessment-release.yml`，只响应 `assessment-v*` 标签
9. Release 上传 ZIP、`.sha256`、变更说明和第三方许可文件

阶段验收：

```bash
cd geo-assessment
bash tools/build-release.sh assessment-v0.1.0
bash tools/verify-release.sh dist/geo-assessment-0.1.0.zip
```

阶段二可以独立合并。CI 建立后，即使暂不发布标签，也能持续保护主分支。

### 阶段三：贡献与安全治理

目标：外部贡献者可以准确选择反馈渠道，维护者具备稳定的版本与安全响应流程。

文件工作：

1. 新增 `geo-assessment/CONTRIBUTING.md`
2. 新增 `geo-assessment/SECURITY.md`，定义受支持版本和私密漏洞报告方式
3. 在根 `.github/ISSUE_TEMPLATE/` 增加测试系统缺陷、题目修订和部署问题模板
4. 增加 Pull Request 模板中的题库来源、许可、数据库迁移和隐私检查项
5. 新增 `geo-assessment/CHANGELOG.md`，采用 Keep a Changelog 结构
6. 发布 `assessment-v0.1.0`，附带安装、升级、回滚和已知限制

阶段验收：

- 三类 Issue 模板能引导用户提供运行版本、复现步骤和许可来源
- 安全问题不会被引导到公开 Issue
- 题目修订必须同时更新答案、解析、原理、来源和内容指纹测试
- Release 页面可下载 ZIP 和 SHA-256，并能从全新环境完成安装

阶段三可以独立合并。社区文件不会改变运行时行为。

## 七、CI 与发布契约

### Pull Request 门禁

```bash
cd geo-assessment
composer validate --strict
composer verify
bash tests/Smoke/http.sh http://127.0.0.1:8080
```

门禁覆盖：

- PHP 语法与类型兼容
- 61 项 PHPUnit 测试
- JavaScript 状态和报告降级测试
- 30 题、100 分、20 单选、10 多选和内容指纹
- PHP 7.3 安装与 SQLite 运行
- 首页、身份、答题、交卷、报告、身份切换和删除记录
- CSRF、安全头、Cookie 与报告归属

### 发布流程

1. 更新 `CHANGELOG.md` 和版本常量
2. 执行完整门禁
3. 构建并隔离安装发布 ZIP
4. 检查 ZIP 不含敏感文件和开发缓存
5. 创建 `assessment-vX.Y.Z` 标签
6. GitHub Actions 生成 Release 并上传校验文件
7. 从 Release 重新下载资产并验证 SHA-256
8. 使用全新目录完成安装冒烟测试
9. 发布在线演示版本并记录对应提交哈希

## 八、安全与隐私门禁

每次提交和发布检查以下模式：

```text
storage/app.sqlite
storage/app.key
storage/backups/
storage/logs/
.env
*.pem
*.key
geo_assessment_session
真实姓名与答题导出文件
```

发布包只保留 `storage/.gitignore`。演示站点的统计脚本、Cookie、数据保存期限和删除方式必须在首页隐私说明中同步披露。

## 九、风险与处理

| 风险 | 处理方式 |
| --- | --- |
| 题目内容的来源边界不清 | 建立题目来源文档，保留论文链接和数据版本，题干与解析采用原创表述 |
| 公开仓库暴露线上部署细节 | 通用配置进入主文档，站点专用配置进入 examples，任何凭据始终排除 |
| PHP 7.3 已结束安全维护 | 继续做兼容测试，README 明确生产环境优先使用受支持 PHP 版本 |
| SQLite 在高并发下写锁竞争 | 保留 WAL、短事务和限流，文档声明单机轻量场景边界 |
| Viewer 与测试系统标签冲突 | 使用 `assessment-v*` 独立标签和独立工作流 |
| 统计脚本引发隐私误解 | 默认关闭，部署者显式配置并承担披露责任 |

## 十、回滚策略

- 仓库变更按阶段独立提交，任一阶段可以通过反向提交撤销
- 阶段一不改变数据库结构，回滚不会触碰用户数据
- 后续数据库迁移必须提供向前修复策略和迁移前备份命令
- Release 采用不可变标签，线上升级前保留代码归档并验证 SQLite 备份
- 新版健康检查失败时恢复上一版代码包，保留现有 `storage/` 目录

## 十一、实施顺序与工作量

| 阶段 | 预计工作量 | 主要产出 |
| --- | ---: | --- |
| 阶段一 | 0.5 至 1 天 | 公开基线、许可与隐私边界、通用部署入口 |
| 阶段二 | 1 至 1.5 天 | CI、可复现构建、Release 自动化 |
| 阶段三 | 0.5 天 | 贡献规范、安全策略、Issue 与 PR 模板 |

完整计划涉及超过 8 个文件和 2 个 GitHub Actions 工作流。每个阶段都可单独合并并维持系统可用。

## 十二、本计划的确认点

推荐方向已经锁定为“先在 GEO Citation Lab 单仓库内开放，稳定后评估拆分”。确认本计划后，从阶段一开始实施；阶段一完成并通过 review 后，再进入 CI 与 Release 自动化。
