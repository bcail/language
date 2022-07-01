import unittest
from lang import TokenType, scan_tokens, parse, evaluate, Var


SOURCE = '(+ 10 2 (- 15 (+ 4 4)) 5)'
EXPECTED_TOKENS = [
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.PLUS},
    {'type': TokenType.NUMBER, 'lexeme': '10'},
    {'type': TokenType.NUMBER, 'lexeme': '2'},
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.MINUS},
    {'type': TokenType.NUMBER, 'lexeme': '15'},
    {'type': TokenType.LEFT_PAREN},
    {'type': TokenType.PLUS},
    {'type': TokenType.NUMBER, 'lexeme': '4'},
    {'type': TokenType.NUMBER, 'lexeme': '4'},
    {'type': TokenType.RIGHT_PAREN},
    {'type': TokenType.RIGHT_PAREN},
    {'type': TokenType.NUMBER, 'lexeme': '5'},
    {'type': TokenType.RIGHT_PAREN},
]

EXPECTED_AST = [
    TokenType.PLUS,
    10,
    2,
    [
        TokenType.MINUS,
        15,
        [
            TokenType.PLUS,
            4,
            4,
        ]
    ],
    5,
]


class ScanTokenTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        tokens = scan_tokens(SOURCE)
        self.assertEqual(tokens, EXPECTED_TOKENS)

    def test_keyword(self):
        tokens = scan_tokens('(= 1 nil)')
        self.assertEqual(tokens,
                [
                    {'type': TokenType.LEFT_PAREN},
                    {'type': TokenType.EQUAL},
                    {'type': TokenType.NUMBER, 'lexeme': '1'},
                    {'type': TokenType.NIL},
                    {'type': TokenType.RIGHT_PAREN},
                ]
            )


class ParseTests(unittest.TestCase):

    def test(self):
        self.maxDiff = None
        ast = parse(EXPECTED_TOKENS)
        self.assertEqual(ast, EXPECTED_AST)

    def test_2(self):
        tokens = [
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.EQUAL},
            {'type': TokenType.TRUE},
            {'type': TokenType.NIL},
            {'type': TokenType.RIGHT_PAREN},
        ]
        ast = parse(tokens)
        self.assertEqual(ast,
                [
                    TokenType.EQUAL,
                    True,
                    None
                ]
            )

    def test_sequence_of_forms(self):
        tokens = [
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.DEF},
            {'type': TokenType.IDENTIFIER, 'lexeme': 'a'},
            {'type': TokenType.NUMBER, 'lexeme': '1'},
            {'type': TokenType.RIGHT_PAREN},
            {'type': TokenType.LEFT_PAREN},
            {'type': TokenType.PLUS},
            {'type': TokenType.IDENTIFIER, 'lexeme': 'a'},
            {'type': TokenType.NUMBER, 'lexeme': '1'},
            {'type': TokenType.RIGHT_PAREN},
        ]
        ast = parse(tokens)
        self.assertEqual(ast,
                [
                    [
                        TokenType.DEF,
                        {'type': TokenType.IDENTIFIER, 'lexeme': 'a'},
                        1
                    ],
                    [
                        TokenType.PLUS,
                        {'type': TokenType.IDENTIFIER, 'lexeme': 'a'},
                        1
                    ]
                ]
            )



class EvalTests(unittest.TestCase):

    def test(self):
        tests = [
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
        ]

        for test in tests:
            with self.subTest(test=test):
                self.assertEqual(evaluate(parse(scan_tokens(test['src']))), test['result'])


if __name__ == '__main__':
    unittest.main()
