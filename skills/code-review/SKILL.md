---
name: code-review
description: 执行深入代码审查，覆盖安全性、性能和可维护性分析。适用于用户要求 review 代码、检查 bug 或审计代码库时。
---

# 代码审查技能

你现在具备执行全面代码审查的专业能力。请遵循下面的结构化方法：

## 审查清单

### 1. 安全性（关键）

检查：
- [ ] **注入漏洞**：SQL、命令、XSS、模板注入
- [ ] **认证问题**：硬编码凭据、弱认证
- [ ] **授权缺陷**：缺少访问控制、IDOR
- [ ] **数据泄露**：日志或错误信息中包含敏感数据
- [ ] **密码学问题**：弱算法、密钥管理不当
- [ ] **依赖项**：已知漏洞（用 `npm audit`、`pip-audit` 检查）

```bash
# 快速安全扫描
npm audit                    # Node.js
pip-audit                    # Python
cargo audit                  # Rust
grep -r "password\|secret\|api_key" --include="*.py" --include="*.js"
```

### 2. 正确性

检查：
- [ ] **逻辑错误**：边界差一、null 处理、边界场景
- [ ] **竞态条件**：并发访问缺少同步
- [ ] **资源泄漏**：文件、连接、内存未关闭或未释放
- [ ] **错误处理**：吞掉异常、缺少错误路径
- [ ] **类型安全**：隐式转换、过度使用 any 类型

### 3. 性能

检查：
- [ ] **N+1 查询**：循环中访问数据库
- [ ] **内存问题**：大对象分配、引用长期保留
- [ ] **阻塞操作**：异步代码中使用同步 I/O
- [ ] **低效算法**：本可 O(n) 却写成 O(n^2)
- [ ] **缺少缓存**：重复执行昂贵计算

### 4. 可维护性

检查：
- [ ] **命名**：清晰、一致、有描述性
- [ ] **复杂度**：函数超过 50 行、嵌套超过 3 层
- [ ] **重复**：复制粘贴的代码块
- [ ] **死代码**：未使用 import、不可达分支
- [ ] **注释**：过时、冗余，或必要处缺失

### 5. 测试

检查：
- [ ] **覆盖率**：关键路径有测试
- [ ] **边界场景**：null、空值、边界值
- [ ] **Mock**：外部依赖被隔离
- [ ] **断言**：检查有意义且具体

## 审查输出格式

```markdown
## Code Review: [file/component name]

### Summary
[1-2 sentence overview]

### Critical Issues
1. **[Issue]** (line X): [Description]
   - Impact: [What could go wrong]
   - Fix: [Suggested solution]

### Improvements
1. **[Suggestion]** (line X): [Description]

### Positive Notes
- [What was done well]

### Verdict
[ ] Ready to merge
[ ] Needs minor changes
[ ] Needs major revision
```

## 常见需要标记的问题模式

### Python
```python
# Bad: SQL 注入
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
# Good:
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# Bad: 命令注入
os.system(f"ls {user_input}")
# Good:
subprocess.run(["ls", user_input], check=True)

# Bad: 可变默认参数
def append(item, lst=[]):  # Bug: 共享同一个可变默认值
# Good:
def append(item, lst=None):
    lst = lst or []
```

### JavaScript/TypeScript
```javascript
// Bad: 原型污染
Object.assign(target, userInput)
// Good:
Object.assign(target, sanitize(userInput))

// Bad: 使用 eval
eval(userCode)
// Good: 永远不要对用户输入使用 eval

// Bad: 回调地狱
getData(x => process(x, y => save(y, z => done(z))))
// Good:
const data = await getData();
const processed = await process(data);
await save(processed);
```

## 审查命令

```bash
# 查看最近变更
git diff HEAD~5 --stat
git log --oneline -10

# 查找潜在问题
grep -rn "TODO\|FIXME\|HACK\|XXX" .
grep -rn "password\|secret\|token" . --include="*.py"

# 检查复杂度（Python）
pip install radon && radon cc . -a

# 检查依赖
npm outdated  # Node
pip list --outdated  # Python
```

## 审查流程

1. **理解上下文**：阅读 PR 描述、关联 issue
2. **运行代码**：尽可能本地构建、测试、运行
3. **自顶向下阅读**：从主入口开始
4. **检查测试**：变更是否有测试？测试是否通过？
5. **安全扫描**：运行自动化工具
6. **人工审查**：使用上面的清单
7. **编写反馈**：具体、给出修复建议、语气友善
