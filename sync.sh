#!/usr/bin/env bash
# sync.sh -- 一键同步 LLM 仓库：commit -> pull --rebase -> push
#
# 用法：
#   ./sync.sh                 交互输入提交说明（直接回车 = 用时间戳）
#   ./sync.sh "改了什么东西"    直接带提交说明，全程无交互
#
# 放在仓库根目录，双击 sync.bat 也可以跑。

set -u
cd "$(dirname "$0")" || exit 1

green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  red "这里不是 git 仓库：$(pwd)"
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "仓库：$(pwd)"
echo "分支：$BRANCH"
echo

# ---------- 情况一：本地没改动，只检查远端 ----------
if [ -z "$(git status --porcelain)" ]; then
  yellow "本地没有改动，检查远端有没有新提交…"
  if git pull --rebase; then
    green "完成，本地已是最新。"
    exit 0
  fi
  red "拉取失败，见上面的报错信息。"
  exit 1
fi

# ---------- 情况二：有改动 ----------
echo "==== 即将同步的改动 ===="
git status --short
echo

# 大文件预警（GitHub 单文件硬上限 100MB，超过就无法推送）
BIG=""
while IFS= read -r f; do
  [ -n "$f" ] || continue
  [ -f "$f" ] || continue
  sz=$(wc -c < "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt 10485760 ] 2>/dev/null; then
    BIG="${BIG}  ${f}  ($((sz / 1048576)) MB)"$'\n'
  fi
done < <(git status --porcelain | sed 's/^...//' | tr -d '"')

if [ -n "$BIG" ]; then
  yellow "警告：以下文件超过 10MB"
  printf '%s' "$BIG"
  printf '仍要提交？(y/N) '
  read -r ans
  case "$ans" in
    y|Y) ;;
    *) echo "已取消，什么都没做。"; exit 1 ;;
  esac
fi

# ---------- 提交说明 ----------
if [ $# -ge 1 ]; then
  MSG="$*"
else
  DEF="$(date '+%Y-%m-%d %H:%M') 自动同步"
  printf '提交说明（直接回车用 "%s"）：' "$DEF"
  read -r MSG
  [ -z "$MSG" ] && MSG="$DEF"
fi

git add -A || { red "git add 失败"; exit 1; }
if ! git commit -m "$MSG" >/dev/null; then
  red "git commit 失败，见上方 git 的报错。"
  exit 1
fi
green "已提交：$MSG"
echo

# ---------- 拉取远端（rebase，保持线性历史）----------
if ! git pull --rebase; then
  echo
  if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
    red "有冲突，脚本停在 rebase 中途。冲突文件："
    git diff --name-only --diff-filter=U
    echo
    echo "手工处理："
    echo "  1) 打开上面的文件，删掉 <<<<<<<  =======  >>>>>>> 这些标记行，留下正确内容"
    echo "  2) git add <文件>"
    echo "  3) git rebase --continue"
    echo "  4) git push"
    echo
    echo "想整体放弃这次 rebase：git rebase --abort"
  else
    red "拉取失败（网络或远端问题）。你的提交还在本地，没丢。"
  fi
  exit 1
fi

# ---------- 推送 ----------
if git push; then
  green "同步完成，已推送到 origin/$BRANCH"
else
  red "推送失败。提交已在本地，等网络恢复后再 git push 即可。"
  exit 1
fi
