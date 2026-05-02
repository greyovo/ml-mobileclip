from wonderwords import RandomWord
import random

rw = RandomWord()

def random_word(part_of_speech="noun"):
    """获取一个指定词性的随机单词"""
    return rw.word(include_parts_of_speech=[part_of_speech])

def random_adjective():
    return rw.word(include_parts_of_speech=["adjective"])

def random_noun():
    return rw.word(include_parts_of_speech=["noun"])

def random_verb():
    return rw.word(include_parts_of_speech=["verb"])

def generate_sentence():
    """从多种句式模板中随机选择一个，生成描述性句子"""

    templates = [
        # "a + 形容词 + 名词" 类
        lambda: f"a {random_adjective()} {random_noun()}",

        # "名词 + 介词 + the + 名词" 类（类似 women by the sea）
        lambda: f"{random_noun()} on the {random_noun()}",
        lambda: f"{random_noun()} by the {random_noun()}",
        lambda: f"{random_noun()} in the {random_noun()}",

        # "a + 名词 + 介词 + the + 名词" 类（类似 apple on the desk）
        lambda: f"a {random_noun()} on a {random_noun()}",
        lambda: f"a {random_noun()} in a {random_noun()}",

        # "a + 形容词 + 名词 + 介词 + the + 名词" 类
        lambda: f"a {random_adjective()} {random_noun()} on the {random_noun()}",
        lambda: f"a {random_adjective()} {random_noun()} in the {random_noun()}",

        # "名词 + 动词-ing" 类
        lambda: f"{random_noun()} {random_verb()}ing",

        # "a + 名词 + 动词-ing + 介词 + the + 名词" 类
        lambda: f"a {random_noun()} {random_verb()}ing by the {random_noun()}",

        # "形容词 + 名词 + 介词 + 形容词 + 名词" 类
        lambda: f"{random_adjective()} {random_noun()} near a {random_adjective()} {random_noun()}",

        # "a + 形容词 + 名词 + with + 名词" 类
        lambda: f"a {random_adjective()} {random_noun()} with {random_noun()}s",
    ]

    template = random.choice(templates)
    return template()

# ---- 生成 10 个随机描述性句子 ----
if __name__ == "__main__":
    print("=" * 50)
    print("  Random Descriptive Sentences (wonderwords)")
    print("=" * 50)
    for i in range(10):
        sentence = generate_sentence()
        print(f"  {i+1:2d}. {sentence}")
    print("=" * 50)
