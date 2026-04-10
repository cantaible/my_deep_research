# Antigravity 代理稳定性排障记录

## 背景

本机使用 `Clash Verge Rev + mihomo` 作为系统代理，`Antigravity` 在代理开启后出现过以下异常：

- `unexpected EOF`
- `stream reading error`
- `Client network socket disconnected before secure TLS connection was established`
- `getaddrinfo ENOTFOUND oauth2.googleapis.com`

本次排障目标不是修改 `Antigravity` 本身，而是在**不新增公网直连暴露面**的前提下，修复代理层导致的网络不稳定问题。

## 现网配置链路

当前实际生效链路如下：

1. `~/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev/profiles.yaml`
2. 当前 active profile: `Lq2BcMaoK0lq`（名字为“我的服务器”）
3. 当前 active script: `profiles/s22aCcDAWNK4.js`
4. mihomo 最终运行配置: `~/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml`

排障前确认到的关键事实：

- `TUN` 已开启
- DNS 为 `fake-ip` 模式
- 原始 active script 为 no-op，没有做任何增强
- Google / Google APIs 域名默认走 `PROXY`

## 根因定位

这次问题不是单一根因，而是以下几个因素叠加：

### 1. TUN + fake-ip 环境下，本地回环与私网流量缺少豁免

原配置没有显式放行：

- `localhost`
- `local`
- `127.0.0.0/8`
- `::1/128`
- `192.168.0.0/16`
- `10.0.0.0/8`
- `172.16.0.0/12`

这类流量一旦被 TUN/fake-ip 接管，容易出现本地回环、解析混乱或链路异常。

### 2. Google 相关服务在代理环境下可能尝试 UDP/QUIC

`Antigravity` 会访问：

- `oauth2.googleapis.com`
- `daily-cloudcode-pa.googleapis.com`
- `cloudcode-pa.googleapis.com`
- 其他 `google.com` / `googleapis.com` 相关域名

在 `TUN + 代理节点` 组合下，UDP/QUIC 链路稳定性通常弱于 TCP。即便节点支持 UDP，多次转发后也更容易出现中断或握手异常。

### 3. 问题主要落在代理层，不是应用逻辑层

排障前，`Antigravity` 本地日志曾出现：

- `getaddrinfo ENOTFOUND oauth2.googleapis.com`
- `Client network socket disconnected before secure TLS connection was established`

这说明当时更接近“解析/连通性失败”，而不是单纯业务鉴权失败。

## 解决思路

修复思路分三层：

### 1. 本地回环与局域网豁免

对本机回环和 RFC1918 私网地址显式 `DIRECT`，避免它们被代理链路再次接管。

### 2. 阻断 Google 相关 UDP，强制回退 TCP

这里不是把 UDP “转换”为 TCP，而是直接 `REJECT` 相关 UDP 请求，迫使上层回退到 HTTPS/TCP。

本次只做最小范围限制：

- `googleapis.com`
- `google.com`

没有扩大到 `DOMAIN-KEYWORD,google`，也没有动 `githubcopilot.com`。

### 3. 对 fake-ip 增加本地域名排除

将本地域名加入 `fake-ip-filter`，避免 `localhost` / `local` / `*.lan` 被 fake-ip 接管。

## 实施方案

为避免直接修改源 YAML 和运行时生成文件，本次修复落在 active script 层：

- 修改文件：`~/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev/profiles/s22aCcDAWNK4.js`

脚本实现原则：

- 保留 `main(config, profileName)` 签名
- 防御性初始化 `config.rules`、`config.dns`、`config.dns["fake-ip-filter"]`
- 以“整体前插”方式注入规则，避免多次 `unshift` 造成顺序反转
- 对 `fake-ip-filter` 做去重追加
- 不新增任何公网 `DIRECT` 规则

### 实际注入规则

本地直连规则：

- `DOMAIN-SUFFIX,localhost,DIRECT`
- `DOMAIN-SUFFIX,local,DIRECT`
- `IP-CIDR,127.0.0.0/8,DIRECT,no-resolve`
- `IP-CIDR6,::1/128,DIRECT,no-resolve`
- `IP-CIDR,192.168.0.0/16,DIRECT,no-resolve`
- `IP-CIDR,10.0.0.0/8,DIRECT,no-resolve`
- `IP-CIDR,172.16.0.0/12,DIRECT,no-resolve`

UDP 限制规则：

- `AND,((NETWORK,UDP),(DOMAIN-SUFFIX,googleapis.com)),REJECT`
- `AND,((NETWORK,UDP),(DOMAIN-SUFFIX,google.com)),REJECT`

`fake-ip-filter` 追加项：

- `+.localhost`
- `+.local`
- `*.lan`

## 验证结果

### 1. Clash Verge 配置成功重生成并通过验证

重启 Clash Verge 后，日志显示：

- `验证成功`
- `配置验证成功`

说明脚本语法和 mihomo 配置均通过。

### 2. 运行配置已包含新增规则

最终生成的 `clash-verge.yaml` 中已出现：

- `dns.fake-ip-filter`
- 本地直连规则
- Google / Google APIs 的 UDP `REJECT` 规则

且这些规则位于原有业务规则之前，优先级正确。

### 3. Google 相关流量恢复为可解析、可连通状态

排障前，`Antigravity` 日志中的错误为：

- `getaddrinfo ENOTFOUND oauth2.googleapis.com`

排障后，同一类请求变为：

- `unauthorized_client`

这说明网络层的“解析/连通性失败”已经消失，请求已经能够到达上游服务，剩余问题变成应用/鉴权层问题。

### 4. 代理侧未再观察到 Google 相关 UDP 继续穿透节点

修复后在 mihomo 服务日志中没有再看到 Google 相关 UDP 被继续送往代理节点，观察到的主要是 TCP 流量。

## 安全边界

本次修复**不会新增公网暴露面**，原因如下：

- 新增 `DIRECT` 只覆盖本地回环和私网地址，不会把公网网站改成直连
- 新增 Google 规则为 `REJECT`，不是 `DIRECT`
- `fake-ip-filter` 只是 DNS 排除，不等于公网直连

需要注意：

- 当前现网本身仍保留历史规则 `GEOSITE,CN,DIRECT`
- 这部分是原配置既有行为，不是本次修复新增

## 回滚方式

如果需要回滚，只需将 `profiles/s22aCcDAWNK4.js` 恢复为 no-op：

```js
function main(config, profileName) {
  return config;
}
```

然后手动重启 Clash Verge，使其重新生成运行配置。

## 后续建议

如果后续仍偶发 `unexpected EOF`，建议按以下顺序继续排查：

1. 持续对照 `Clash Verge service_latest.log` 与 `Antigravity cloudcode.log`
2. 如仍出现 Google 相关异常，可将 UDP 限制范围扩大到 `gstatic.com`
3. 若仍不稳定，再评估是否需要扩大到更广的 Google 域名集合
4. 在没有必要之前，不建议直接扩大到 `DOMAIN-KEYWORD,google`

## 结论

本次修复的本质是：在不改 `Antigravity` 的情况下，对 `Clash Verge` 的 TUN/fake-ip 行为做最小范围纠偏。

从结果看，当前已经实现了两个目标：

- 代理配置稳定性增强
- `Antigravity` 从“网络层失败”转为“应用层鉴权失败”

因此，这次修复可以判定为**网络层有效**。
