from lark import Lark, Transformer

parser = Lark('''
  %import common.WS
  %ignore WS

  COMMENT : "(*" /.+/ "*)"
  %ignore COMMENT

  NUMBER : /-?\d+(\.\d+)?/

  ?expr : arith
  ?arith : times (/[+-]/ times)*
  ?times : dot dot*
  ?dot : value ("." value)*
  ?value : negative | array | paren | variable | NUMBER
  negative : "-" value
  array : "{" expr ("," expr)* "}"
  paren : "(" expr ")"
  variable : /[a-zA-Z][a-zA-Z0-9]*/ [ "[" expr "]" ]
''', start='expr')

class ToPython(Transformer):
    def arith(self, items):
        return '(' + ''.join(items) + ')'
    def times(self, items):
        return '(' + '*'.join(items) + ')'
    def dot(self, items):
        return 'mdot([' + ','.join(items) + '])'
    def negative(self, items):
        return '(-' + ''.join(items) + ')'
    def array(self, items):
        return 'np.array([' + ','.join(items) + '])'
    def paren(self, items):
        return '(' + ''.join(items) + ')'
    def variable(self, items):
        name = items[0]
        if items[1] is not None:
            name += '[' + items[1] + ']'
        return name

def convert(string):
    return ToPython().transform(parser.parse(string)).replace("I","1j")

print(convert(''' {-4 F0 . dt0 . q0 - 2 q0 . dt0 . q0 +
  2 (-I + 2 F0 . q0 + q0 . q0) (2 F0 . dt0 . q0 +
     2 q0 . dt0 . q0), -4 F0 . dt0 . q0 - 2 F0 . dt1 . q1 -
  4 F3 . dt0 . q0 - 2 F3 . dt1 . q1 - 4 q0 . dt0 . q0 -
  2 q0 . dt1 . q1 + 4 q2 . dt0 . q0 +
  2 (-I - F0 . F0 - 2 F0 . q0 + F3 . F3 - 2 F3 . q2 - q0 . q0 +
     q2 . q2) (-2 F0 . dt0 . q0 + 2 F3 . dt0 . q0 + 2 F3 . dt1 . q1 -
     2 q0 . dt0 . q0 - 2 q2 . dt0 . q0 - 2 q2 . dt1 . q1) +
  2 q2 . dt1 . q1, -4 F3 . dt0 . q0 - 4 F3 . dt1 . q1 +
  2 q2 . dt0 . q0 + 2 q2 . dt1 . q1 +
  2 (-I + 2 F3 . q2 - q2 . q2) (-2 F3 . dt0 . q0 - 2 F3 . dt1 . q1 +
     2 q2 . dt0 . q0 + 2 q2 . dt1 . q1)} '''))

print()

print(convert('''
    {{-4 F0 . R[l] . q0 - 2 q0 . R[l] . q0 +
   2 (-I + 2 F0 . q0 + q0 . q0) (2 F0 . R[l] . q0 +
      2 q0 . R[l] . q0), -4 F0 . R[l] . q0 - 4 F3 . R[l] . q0 -
   4 q0 . R[l] . q0 +
   2 (-I - F0 . F0 - 2 F0 . q0 + F3 . F3 - 2 F3 . q2 - q0 . q0 +
      q2 . q2) (-2 F0 . R[l] . q0 + 2 F3 . R[l] . q0 -
      2 q0 . R[l] . q0 - 2 q2 . R[l] . q0) +
   4 q2 . R[l] . q0, -4 F3 . R[l] . q0 + 2 q2 . R[l] . q0 +
   2 (-I + 2 F3 . q2 - q2 . q2) (-2 F3 . R[l] . q0 +
      2 q2 . R[l] . q0),
  R[l] . q0}, {0, -2 F0 . R[l] . q1 - 2 F3 . R[l] . q1 -
   2 q0 . R[l] . q1 +
   2 (-I - F0 . F0 - 2 F0 . q0 + F3 . F3 - 2 F3 . q2 - q0 . q0 +
      q2 . q2) (2 F3 . R[l] . q1 - 2 q2 . R[l] . q1) +
   2 q2 . R[l] . q1, -4 F3 . R[l] . q1 + 2 q2 . R[l] . q1 +
   2 (-I + 2 F3 . q2 - q2 . q2) (-2 F3 . R[l] . q1 +
      2 q2 . R[l] . q1), R[l] . q1}, {0, 0, 0, R[l] . q2}}
    '''))

print()

print(convert('''
   {{-4 F0 . E3[l] . q0 + 
   4 ((1 - I) + 2 F0 . q0) F0 . E3[l] . q0, -4 F0 . E3[l] . q0 - 
   4 F3 . E3[l] . q0 + 
   2 (-I - F0 . F0 - 2 F0 . q0 + F3 . F3 - 
      2 F3 . q2) (-2 F0 . E3[l] . q0 + 2 F3 . E3[l] . q0 - 
      2 q2 . E3[l] . q0) + 4 q2 . E3[l] . q0, -4 F3 . E3[l] . q0 + 
   2 q2 . E3[l] . q0 + 
   2 ((-1 - I) + 2 F3 . q2) (-2 F3 . E3[l] . q0 + 
      2 q2 . E3[l] . q0), {Ort[0] . E3[l] . q0, Ort[1] . E3[l] . q0, 
   Ort[2] . E3[l] . q0}}, {0, -2 F0 . E3[l] . q1 - 
   2 F3 . E3[l] . q1 - 2 q0 . E3[l] . q1 + 
   2 (-I - F0 . F0 - 2 F0 . q0 + F3 . F3 - 
      2 F3 . q2) (2 F3 . E3[l] . q1 - 2 q2 . E3[l] . q1) + 
   2 q2 . E3[l] . q1, -4 F3 . E3[l] . q1 + 2 q2 . E3[l] . q1 + 
   2 ((-1 - I) + 2 F3 . q2) (-2 F3 . E3[l] . q1 + 
      2 q2 . E3[l] . q1), {Ort[0] . E3[l] . q1, Ort[1] . E3[l] . q1, 
   Ort[2] . E3[l] . q1}}, {0, 0, 
  0, {Ort[0] . E3[l] . q2, Ort[1] . E3[l] . q2, Ort[2] . E3[l] . q2}}}
    '''))
