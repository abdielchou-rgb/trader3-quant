"""
3号交易员 — 因子表达式解析器

将字符串表达式解析为表达式树（Node）。

支持语法:
    add(close, delay(close, 5))
    ts_mean(volume, 20)
    rank(close)
    mul(sub(close, ts_mean(close, 20)), div(volume, ts_mean(volume, 20)))
    const(2.5)
"""

from __future__ import annotations

import re
from typing import List, Optional

from .gp import FIELDS, CONSTANTS, Node, OPS


_TOKEN_RE = re.compile(r"""
    \s*
    (?P<name>[a-zA-Z_][a-zA-Z0-9_]*)   # 函数名/字段名
    |\s*
    (?P<num>-?\d+\.?\d*)                # 数字
    |\s*
    (?P<lparen>\()
    |\s*
    (?P<rparen>\))
    |\s*
    (?P<comma>,)
""", re.VERBOSE)


def _tokenize(s: str) -> List[tuple]:
    tokens = []
    pos = 0
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            # 跳过空白
            if s[pos].isspace():
                pos += 1
                continue
            raise ValueError(f"无法解析字符 '{s[pos]}' 在位置 {pos}")
        pos = m.end()
        if m.lastgroup == "name":
            tokens.append(("name", m.group("name")))
        elif m.lastgroup == "num":
            tokens.append(("num", float(m.group("num"))))
        elif m.lastgroup == "lparen":
            tokens.append(("lparen", "("))
        elif m.lastgroup == "rparen":
            tokens.append(("rparen", ")"))
        elif m.lastgroup == "comma":
            tokens.append(("comma", ","))
    return tokens


class _Parser:
    def __init__(self, tokens: List[tuple]):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        t = self.peek()
        self.pos += 1
        return t

    def parse(self) -> Node:
        node = self.parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError("表达式结尾有多余 token")
        return node

    def parse_expr(self) -> Node:
        return self.parse_function_or_field()

    def parse_function_or_field(self) -> Node:
        tok = self.next()
        if tok is None:
            raise ValueError("表达式为空")
        kind, val = tok

        if kind == "num":
            return Node(op="const", value=val)

        if kind == "name":
            # 可能是字段或函数
            next_tok = self.peek()
            if next_tok and next_tok[0] == "lparen":
                return self.parse_function(val)
            # 字段或常数
            if val in FIELDS:
                return Node(op=val)
            if val in ("const", "c"):
                # 处理 const(数字)
                # 实际上是函数，但需要特殊处理
                # const(...) 应已被 parse_function 捕获，这里做兜底
                return Node(op="const", value=0.0)
            raise ValueError(f"未知字段: {val}")

        raise ValueError(f"意外的 token: {kind}")

    def parse_function(self, name: str) -> Node:
        # 消费 '('
        self.next()  # lparen

        if name in ("const", "c"):
            # const(数字)
            num_tok = self.next()
            if num_tok[0] != "num":
                raise ValueError("const 需要数字参数")
            self._expect_rparen()
            return Node(op="const", value=num_tok[1])

        if name not in OPS:
            raise ValueError(f"未知操作符: {name}")

        n_args = OPS[name][0]
        children = []
        for i in range(n_args):
            if i > 0:
                comma = self.next()
                if comma and comma[0] != "comma":
                    raise ValueError("缺少逗号")
            children.append(self.parse_expr())

        self._expect_rparen()
        return Node(op=name, children=children)

    def _expect_rparen(self):
        tok = self.next()
        if tok is None or tok[0] != "rparen":
            raise ValueError("缺少右括号")


def parse_expr(s: str) -> Node:
    """解析表达式字符串为 Node。"""
    if isinstance(s, Node):
        return s
    tokens = _tokenize(s)
    parser = _Parser(tokens)
    return parser.parse()


if __name__ == "__main__":
    for expr in [
        "add(close, delay(close, 5))",
        "ts_mean(volume, 20)",
        "rank(close)",
        "mul(sub(close, ts_mean(close, 20)), div(volume, ts_mean(volume, 20)))",
        "const(2.5)",
        "ts_corr(close, volume, 10)",
    ]:
        try:
            node = parse_expr(expr)
            print(f"{expr}  =>  {node.to_str()}")
        except Exception as e:
            print(f"{expr}  =>  ERROR: {e}")