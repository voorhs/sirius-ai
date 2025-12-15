---
marp: true
math: true
paginate: true
---

### Алгоритмы и анализ данных, 2025

# Классические методы векторизации текстов

Алексеев Илья

---

# Обработка естественного языка

- векторизация текста
- предобработка текста
- регулярные выражения

---

## Векторизация

![](figures/vectorization.jpg)

---

## Векторизация

1. Базовые подходы:
    * **Мешок слов (Bag Of Words)**
    * **Tf-Idf**
2. Матричные разложения:
    * LSA/LDA
    * BigARTM
3. Нейросетевые подходы:
    * Word2Vec (Skip-gram, CBoW, FastText, Glove, ...)
    * BERT, sentence embeddings

---

## Что такое текст?

Обучающая коллекция документов (текстов):
$$D = \{d_1, d_2 \ldots d_N \}$$

Документ:
$$d_i = (w_1, w_2, \ldots w_n),$$
где $w_i$ — токен (слово) из вокабулярия (словаря) $V$.

---

Токенизация — разделение текста на токены, элементарные единицы текста

В большинстве случае токен это слово!

Если пользоваться методом `.split()`, токен — последовательность букв, разделённая пробельными символами

Можно использовать регулярные выражения и модули `re`, `regex`.

Можно использовать специальные токенизаторы, например из
`nltk`:
- `RegexpTokenizer`
- `BlanklineTokenizer`
- И ещё около десятка штук

---

```python
from nltk.tokenize import word_tokenize

example = 'Но не каждый хочет что-то исправлять:('
word_tokenize(example, language='russian')
```
```
['Но', 'не', 'каждый', 'хочет', 'что-то', 'исправлять', ':(']
```

---

```python
from nltk.tokenize import sent_tokenize

sent = 'Hey! Is Mr. Bing waiting for you?'
nltk.tokenize.sent_tokenize(sent)
```
```
['Hey!', 'Is Mr. Bing waiting for you?']
```

---

## Bag of Words (Count Vectorizer)

Предположим:
* Порядок токенов в тексте не важен
* Важно лишь сколько раз токен $w$ входит в текст $d$

Term-frequency, число вхождений слова в текст: $\text{tf}(w, d)$

Векторизация:
$$v(d) = (\text{tf}(w_{i}, d))_{i=1}^{|V|} $$

---

## Bag of Words (Count Vectorizer)

```python
from sklearn.feature_extraction.text import CountVectorizer

s = [
    'my name is Joe',
    'your name are Joe',
    'my father is Joe'
]
vectorizer = CountVectorizer()
vectorizer.fit_transform(s).toarray()
```

```
array([[0, 0, 1, 1, 1, 1, 0],
       [1, 0, 0, 1, 0, 1, 1],
       [0, 1, 1, 1, 1, 0, 0]])
```

---

## Проблемы

- Нет учёта контекста и порядка слов
- Учет слов, которые не несут дискриминативной информации
- Огромное признаковое пространство
