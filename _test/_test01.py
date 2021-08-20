
tuple_example = (['Mourinho', 'a'], ['Herrera', 'b'])

# print(tuple_example[0][1]) # a
tuple_example[0][1] = 'c'
# print(tuple_example[0][1]) # c

# print(tuple_example[0]) # ['Mourinho', 'c']
# tuple_example[0] = ['McTominay', 'd'] 
# TypeError: 'tuple' object does not support item assignment



dict_example = {'Son' : 'TotH', 'DeBruyne' : 'ManC', 'Pogba' : 'ManU', 'Messi' : 'PSG'}

# print(dict_example['Messi']) # PSG
dict_example['Messi'] = 'ManC'
# print(dict_example['Messi']) # ManC

# print(dict_example.items())
# [('Son', 'TotH'), ('DeBruyne', 'ManC'), ('Pogba', 'ManU'), ('Messi', 'ManC')]
dict_example['Chae'] = 'ManU'
# print(dict_example.items())
# [('Son', 'TotH'), ('DeBruyne', 'ManC'), ('Pogba', 'ManU'), ('Messi', 'ManC'), ('Chae', 'ManU')]



set_example1 = { 'm', 'e', 's', 's', 'i'}
print(set_example1)  # {'m', 's', 'i', 'e'}

set_example1.add(30)
print(set_example1)  # {'m', 's', 'i', 'e', '30'}

set_example2 = { 's', 'o', 'n'}
intersection = set_example1.intersection(set_example2)
difference = set_example1.difference(set_example2)
print(intersection) # {'s'}
print(difference)   # {'i', 'e', 30, 'm'}







