"""Hello World 示例模块。

演示基础的命令行输出、类型注解、文档字符串以及 main 入口保护。
"""


def greet(name: str) -> None:
    """打印一条问候消息。

    Args:
        name: 被问候的人的名称。
    """
    print(f"Hello, {name}")


if __name__ == "__main__":
    greet("Claude")
