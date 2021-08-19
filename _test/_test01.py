
tuple_example = (['Mourinho', 'a'], ['Herrera', 'b'])

# print(tuple_example[0][1]) # a
tuple_example[0][1] = 'c'
# print(tuple_example[0][1]) # c

# print(tuple_example[0]) # ['Mourinho', 'c']
# tuple_example[0] = ['McTominay', 'd'] 
# TypeError: 'tuple' object does not support item assignment



dict_example = {'Son' : 'TotH', 'DeBruyne' : 'ManC', 'Pogba' : 'ManU', 'Messi' : 'PSG'}

print(dict_example['Messi']) # PSG
dict_example['Messi'] = 'ManC'
print(dict_example['Messi']) # ManC

print(dict_example.items())
# [('Son', 'TotH'), ('DeBruyne', 'ManC'), ('Pogba', 'ManU'), ('Messi', 'ManC')]
dict_example['Chae'] = 'ManU'
print(dict_example.items())
# [('Son', 'TotH'), ('DeBruyne', 'ManC'), ('Pogba', 'ManU'), ('Messi', 'ManC'), ('Chae', 'ManU')]


