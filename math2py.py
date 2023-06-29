from lark import Lark, Transformer

parser = Lark('''
  %import common.WS
  %ignore WS

  COMMENT : "(*" /.+/ "*)"
  %ignore COMMENT

  NUMBER : /\d+(\.\d+)?/

  ?expr : arith
  ?arith : times (/[+-]/ times)*
  ?times : dot dot*
  ?dot : value ("." value)*
  ?value : array | paren | variable | NUMBER | negative
  negative : "-" value
  array : "{" expr ("," expr)* "}"
  paren : "(" expr ")"
  variable : /[a-zA-Z][a-zA-Z0-9]*/ [ "[" expr "]" ]
''', start='expr', ambiguity='explicit')

class Disambiguate(Transformer):
    def _ambig(self, items):
        return max(items, key=lambda t: len(t.children))

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
        if name == 'I':
            name = '1j'
            if items[1] is not None:
                raise ValueError("You can't index into I.")
        elif items[1] is not None:
            name += '[' + items[1] + ']'
        return name

def convert(string):
    return ToPython().transform(Disambiguate().transform(parser.parse(string)))
print(
convert('''
{{z^2 -> (-2 I3 + I4 - I4 x^2 - I4 y^2 - 
    2 Sqrt[I3^2 - I3 I4 - I1 I4 x^2 + I3 I4 x^2 - I2 I4 y^2 + 
      I3 I4 y^2])/
   I4}, {z^2 -> (-2 I3 + I4 - I4 x^2 - I4 y^2 + 
    2 Sqrt[I3^2 - I3 I4 - I1 I4 x^2 + I3 I4 x^2 - I2 I4 y^2 + 
      I3 I4 y^2])/I4}}
 ''')
)

# math2py.py
# Displaying math2py.py.