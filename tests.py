import unittest
from lang import TokenType, scan_tokens, parse, evaluate, Keyword, Symbol, Var, Vector


SOURCE = '(+ 10 2 (- 15 (+ 4 4)) -5)'
EXPECTED_TOKENS = [
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '+'},
    {'type': TokenType.NUMBER, 'lexeme': '10'},
    {'type': TokenType.NUMBER, 'lexeme': '2'},
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '-'},
    {'type': TokenType.NUMBER, 'lexeme': '15'},
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.SYMBOL, 'lexeme': '+'},
    {'type': TokenType.NUMBER, 'lexeme': '4'},
    {'type': TokenType.NUMBER, 'lexeme': '4'},
    {'type': TokenType.RIGHT_PAREN},
    {'type': TokenType.RIGHT_PAREN},
    {'type': TokenType.NUMBER, 'lexeme': '-5'},
    {'type': TokenType.RIGHT_PAREN},
]

EXPECTED_AST_FORMS = [
    [
        Symbol('+'),
        10,
        2,
        [
            Symbol('-'),
            15,
            [
                Symbol('+'),
                4,
                4,
            ]
        ],
        -5,
    ]
]


class ScanTokenTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        tokens = scan_tokens(SOURCE)
        self.assertEqual(tokens, EXPECTED_TOKENS)

    def test_nil(self):
        tokens = scan_tokens('(= 1 nil)')
        self.assertEqual(tokens,
                [
                    {'type': TokenType.LEFT_PAREN},
                    {'type': TokenType.SYMBOL, 'lexeme': '='},
                    {'type': TokenType.NUMBER, 'lexeme': '1'},
                    {'type': TokenType.NIL},
                    {'type': TokenType.RIGHT_PAREN},
                ]
            )

    def test_number(self):
        tokens = scan_tokens('2')
        self.assertEqual(tokens, [{'type': TokenType.NUMBER, 'lexeme': '2'}])


class ParseTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        ast = parse(EXPECTED_TOKENS)
        self.assertEqual(ast.forms, EXPECTED_AST_FORMS)

    def test_2(self):
        tokens = [
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': '='},
            {'type': TokenType.TRUE},
            {'type': TokenType.NIL},
            {'type': TokenType.RIGHT_PAREN},
        ]
        ast = parse(tokens)
        self.assertEqual(ast.forms,
            [
                [
                    Symbol('='),
                    True,
                    None
                ]
            ]
        )

    def test_sequence_of_forms(self):
        tokens = [
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': 'def'},
            {'type': TokenType.SYMBOL, 'lexeme': 'a'},
            {'type': TokenType.NUMBER, 'lexeme': '1'},
            {'type': TokenType.RIGHT_PAREN},
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.SYMBOL, 'lexeme': '+'},
            {'type': TokenType.SYMBOL, 'lexeme': 'a'},
            {'type': TokenType.NUMBER, 'lexeme': '1'},
            {'type': TokenType.RIGHT_PAREN},
        ]
        ast = parse(tokens)
        self.assertEqual(ast.forms,
                [
                    [
                        Symbol('def'),
                        Symbol('a'),
                        1
                    ],
                    [
                        Symbol('+'),
                        Symbol('a'),
                        1
                    ]
                ]
            )


class EvalTests(unittest.TestCase):

    def test(self):
        tests = [
            {'src': '2', 'result': 2},
            {'src': 'nil', 'result': None},
            {'src': 'true', 'result': True},
            {'src': 'false', 'result': False},
            {'src': ':hello', 'result': Keyword(':hello')},
            {'src': '"hello"', 'result': 'hello'},
            {'src': '"hello "', 'result': 'hello '},
            {'src': '(quote (1 2))', 'result': [1, 2]},
            {'src': '[1 2]', 'result': Vector([1, 2])},
            {'src': '{1 5, 2 4}', 'result': {1: 5, 2: 4}},
            {'src': '1 2 nil', 'result': [1, 2, None]}, #multiple forms, not a list
            {'src': '1 [1 2] nil', 'result': [1, Vector([1, 2]), None]}, #multiple forms, not a list
            {'src': '(+ 1 2)', 'result': 3},
            {'src': '(+ 1 (+ 1 1))', 'result': 3},
            {'src': '(+ 1 (- 4 2))', 'result': 3},
            {'src': '(* 2 (- 4 2) 3)', 'result': 12},
            {'src': '(/ 12 (- 4 2) 3)', 'result': 2},
            {'src': '(+ 10 -2)', 'result': 8},
            {'src': '(= 1 2)', 'result': False},
            {'src': '(= 1 1 1)', 'result': True},
            {'src': '(= true 2)', 'result': False},
            {'src': '(= true nil)', 'result': False},
            {'src': '(if true true false)', 'result': True},
            {'src': '(if false true false)', 'result': False},
            {'src': '(def a 1)', 'result': Var(name='a', value=1)},
            {'src': '(def a 1) (+ a 2)', 'result': [Var(name='a', value=1), 3]},
            {'src': '(let [x 1] x)', 'result': 1},
            {'src': '((fn [x] x) 1)', 'result': 1},
            {'src': '(let [x 1 y 2] (+ x y))', 'result': 3},
            {'src': '((fn [x y] (+ x y)) 1 2)', 'result': 3},
            {'src': '(str)', 'result': ''},
            {'src': '(str 1)', 'result': '1'},
            {'src': '(str "hello " "world")', 'result': 'hello world'},
            {'src': '(str/split "hello world")', 'result': ['hello', 'world']},
            {'src': '(get {1 2} 1)', 'result': 2},
            {'src': '(keys {1 2 "a" 3})', 'result': [1, 'a']},
            {'src': '(vals {1 2 "a" 3})', 'result': [2, 3]},
            {'src': '(contains? {1 2 "a" 3} 1)', 'result': True},
            {'src': '(contains? {1 2 "a" 3} "not-found")', 'result': False},
            {'src': '(assoc {1 2 "a" 3} "new-key" "new-val")', 'result': {1: 2, 'a': 3, 'new-key': 'new-val'}},
            {'src': '(dissoc {1 2 "a" 3} 1)', 'result': {'a': 3}},
            {'src': '(println "hello")', 'result': None},
        ]

        for test in tests:
            with self.subTest(test=test):
                print(f'*** test: {test["src"]}')
                self.assertEqual(parse(scan_tokens(test['src'])).evaluate(), test['result'])

    def test_other(self):
        results = parse(scan_tokens('(def f1 (fn [x y] (+ x y))) (f1 1 2)')).evaluate()
        self.assertEqual(results[1], 3)

        results = parse(scan_tokens('(defn f1 [x y] (+ x y)) (f1 1 2)')).evaluate()
        self.assertEqual(results[1], 3)

    def test_exceptions(self):
        with self.assertRaises(Exception) as cm:
            parse(scan_tokens('(1 2 3)')).evaluate()
        self.assertIn('not callable', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
