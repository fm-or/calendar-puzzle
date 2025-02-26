from __future__ import annotations
from gurobipy import *
from typing import Tuple, List
from operator import itemgetter
from pulp import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import Grid
from math import ceil, sqrt


def date_to_coords(day: int, month: 'int') -> List[Tuple[int]]:
    day_x = (day-1)%7
    day_y = 4-int((day-1)/7)
    month_x = (month-1)%6
    month_y = 6 - int((month-1)/6)
    return [(day_x, day_y), (month_x, month_y)]

def month_to_text(month: 'int') -> str:
    return ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'][month-1]


class OrientedForm():

    def __init__(self, coords):
        self._coords = coords

    def rotate_clockwise(self) -> OrientedForm:
        coords_r = list()
        max_x = max(self._coords, key=itemgetter(0))[0]
        for x, y in self._coords:
            coords_r.append((y, max_x - x))
        return OrientedForm(tuple(coords_r))

    def flip_horizontal(self) -> OrientedForm:
        coords_r = list()
        max_x = max(self._coords, key=itemgetter(0))[0]
        for x, y in self._coords:
            coords_r.append((max_x - x, y))
        return OrientedForm(tuple(coords_r))

    def _flip_vertical(self) -> OrientedForm:
        coords_r = list()
        max_y = max(self._coords, key=itemgetter(1))[1]
        for x, y in self._coords:
            coords_r.append((x, max_y - y))
        return OrientedForm(tuple(coords_r))

    def __eq__(self, other): 
        if not isinstance(other, OrientedForm):
            return NotImplemented
        for coord in self._coords:
            if coord not in other._coords:
                return False
        for coord in other._coords:
            if coord not in self._coords:
                return False
        return True

    def __str__(self):
        max_x = max(self._coords, key=itemgetter(0))[0]
        max_y = max(self._coords, key=itemgetter(1))[1]
        lines = list()
        for y in range(max_y+1):
            line = ""
            for x in range(max_x+1):
                if (x, max_y-y) in self._coords:
                    line += "#"
                else:
                    line += " "
            lines.append(line)
        return "\n".join(lines)

    def __hash__(self):
        return hash(self._coords)
        

class Form():

    def __init__(self, coords: Tuple[Tuple[int]]):
        self._set_all_orientations(OrientedForm(coords))

    def _set_all_orientations(self, form: OrientedForm) -> None:
        self._orientations = list()
        for i in range(4):
            form_h = form.flip_horizontal()
            form_v = form.flip_horizontal()
            if form not in self._orientations:
                self._orientations.append(form)
            if form_h not in self._orientations:
                self._orientations.append(form_h)
            if form_v not in self._orientations:
                self._orientations.append(form_v)
            form = form.rotate_clockwise()
                 
    def get_all_orientations(self) -> List[OrientedForm]:
        return self._orientations.copy()
        

start_time = time()
forms = list()
forms.append(Form(((0, 0), (1, 0), (1, 1), (2, 1), (3, 1))))
forms.append(Form(((0, 0), (1, 0), (2, 0), (3, 0), (2, 1))))
forms.append(Form(((0, 0), (0, 1), (1, 1), (2, 1), (2, 2))))
forms.append(Form(((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))))
forms.append(Form(((0, 0), (1, 0), (2, 0), (3, 0), (3, 1))))
forms.append(Form(((0, 0), (1, 0), (2, 0), (0, 1), (0, 2))))
forms.append(Form(((0, 0), (1, 0), (2, 0), (0, 1), (1, 1))))
forms.append(Form(((0, 0), (1, 0), (2, 0), (0, 1), (2, 1))))

size = (7, 7)
date = (25, 9)
forbidden = [(3, 0), (4, 0), (5, 0), (6, 0), (6, 5), (6, 6)]
forbidden += date_to_coords(*date)

prob = LpProblem(sense=LpMaximize)
z = dict()
for form in forms:
    for form_o in form.get_all_orientations():
        max_x = max(form_o._coords, key=itemgetter(0))[0]
        max_y = max(form_o._coords, key=itemgetter(1))[1]
        for x in range(size[0]-max_x):
            for y in range(size[1]-max_y):
                z[form_o, x, y] = LpVariable(name="x"+str(form_o._coords)+str(x)+str(y), cat='Binary')

prob += lpSum(z)

# prob += lpSum(z[form_o, 0, 5] for form_o in forms[3].get_all_orientations() if (form_o, 0, 5) in z) == 1
# prob += lpSum(z[form_o, 3, 5] for form_o in forms[6].get_all_orientations() if (form_o, 3, 5) in z) == 1

# exactly one orientation on a position per form
for form in forms:
    prob += lpSum(z[form_o, x, y] for form_o in form.get_all_orientations() for x in range(size[0]-max(form_o._coords, key=itemgetter(0))[0]) for y in range(size[1]-max(form_o._coords, key=itemgetter(1))[1])) == 1

# each selected oriented form on a position blocks the respective fields
for x in range(size[0]):
    for y in range(size[1]):
        relevant_form_coords = set()
        for form in forms:
            for form_o in form.get_all_orientations():
                max_x = max(form_o._coords, key=itemgetter(0))[0]
                max_y = max(form_o._coords, key=itemgetter(1))[1]
                for xf in range(min(x+1, size[0]-max_x)):
                    for yf in range(min(y+1, size[1]-max_y)):
                        for dx, dy in form_o._coords:
                            if xf + dx == x and yf + dy == y:
                                relevant_form_coords.add((form_o, xf, yf))
        prob += lpSum(z[form_o, xf, yf] for form_o, xf, yf in relevant_form_coords) == (0 if (x, y) in forbidden else 1)

# Löse die Probleminstanz mit CBC
#solver = PULP_CBC_CMD(msg=0)
solver = GUROBI_CMD(msg=0)
prob.solve(solver=solver)

solutions = list()
i = 0
while LpStatus[prob.status] == "Optimal":
    # Store solution
    solution = [[None for y in range(size[1])] for x in range(size[0])]
    for form in forms:
        for form_o in form.get_all_orientations():
            max_x = max(form_o._coords, key=itemgetter(0))[0]
            max_y = max(form_o._coords, key=itemgetter(1))[1]
            for x in range(size[0]-max_x):
                for y in range(size[1]-max_y):
                    if z[form_o, x, y].varValue > 0.999:
                        for xf, yf in form_o._coords:
                            solution[x+xf][y+yf] = form_o
    solutions.append(solution)
    print("Found", i+1, "solutions in", round(time()-start_time, 2), "seconds.           ", end="\r")

    # Forbid solution
    prob += lpSum(z[form_o, x, y] for form in forms for form_o in form.get_all_orientations() for x in range(size[0]-max(form_o._coords, key=itemgetter(0))[0]) for y in range(size[1] - max(form_o._coords, key=itemgetter(1))[1]) if z[form_o, x, y].varValue > 0.999) <= len(forms) - 1
    prob.solve(solver=solver)
    i += 1
print("")

start_time = time()
cols = math.ceil(sqrt(len(solutions)*16/9))
rows = int((len(solutions)/cols)+0.99)
width = 0.9
fig = plt.figure(str(len(solutions))+" solutions for " + str(date[0]) + " " + month_to_text(date[1]))
grid = Grid(fig, rect=111, nrows_ncols=(rows, cols), axes_pad=0.05)
#fig, axs = plt.subplots(rows, cols)
#for row in range(rows):
#    for col in range(cols):
#        k = row*cols + col
#        ax = axs[row, col]
for k, ax in enumerate(grid):
    if True:
        if k < len(solutions):
            solution = solutions[k]    
            ax.add_patch(Rectangle((-0.55, -0.55), size[0]+0.55, size[1]+0.55, color='k', linewidth=0))
            ax.add_patch(Rectangle((-0.5, -0.5), size[0], size[1], color='w', linewidth=0))
            for x in range(size[0]):
                for y in range(size[1]):
                    if (x, y) not in forbidden:
                        if x+1 < size[0] and (x+1, y) not in forbidden and solution[x][y] == solution[x+1][y]:
                            ax.add_patch(Rectangle((x-width/2, y-width/2), 1+width, width, linewidth=0))
                        if y+1 < size[1] and (x, y+1) not in forbidden and solution[x][y] == solution[x][y+1]:
                            ax.add_patch(Rectangle((x-width/2, y-width/2), width, 1+width, linewidth=0))
            
            for x in range(size[0]-1):
                for y in range(size[1]-1):
                    square = {solution[x][y], solution[x][y+1], solution[x+1][y], solution[x+1][y+1]}
                    if len(square) == 1:
                        ax.add_patch(Rectangle((x, y), 1, 1, linewidth=0))
            for i in range(len(forbidden)-1):
                for j in range(i+1, len(forbidden)):
                    if forbidden[i] not in date_to_coords(*date) and forbidden[j] not in date_to_coords(*date):
                        xi, yi = forbidden[i]
                        xj, yj = forbidden[j]
                        if abs(xi - xj) + abs(yi - yj) == 1:
                            ax.add_patch(Rectangle((min(xi, xj)-width/2, min(yi, yj)-width/2), abs(xi-xj)+width, abs(yi-yj)+width, linewidth=0, color='gray'))
            ax.add_patch(Rectangle(tuple(map(sum, zip(date_to_coords(*date)[0], (-width/2, -width/2)))), width, width, color='k', linewidth=0))
            ax.add_patch(Rectangle(tuple(map(sum, zip(date_to_coords(*date)[0], (-(width-0.06)/2, -(width-0.06)/2)))), width-0.06, width-0.06, color='w', linewidth=0))
            ax.text(*date_to_coords(*date)[0], date[0], size='xx-small', weight='bold', verticalalignment='center', horizontalalignment='center')
            ax.add_patch(Rectangle(tuple(map(sum, zip(date_to_coords(*date)[1], (-width/2, -width/2)))), width, width, color='k', linewidth=0))
            ax.add_patch(Rectangle(tuple(map(sum, zip(date_to_coords(*date)[1], (-(width-0.06)/2, -(width-0.06)/2)))), width-0.06, width-0.06, color='w', linewidth=0))
            ax.text(*date_to_coords(*date)[1], month_to_text(date[1]), size='xx-small', weight='bold', verticalalignment='center', horizontalalignment='center')
        ax.set_xlim(-0.55, size[0]-0.45)
        ax.set_ylim(-0.55, size[1]-0.45)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.title.set_visible(False)

#plt.axis('off')
fig.tight_layout()
print("Generated plots in", round(time()-start_time, 2), "seconds.")
plt.show()    

'''for form in forms:
    for form_o in form.get_all_orientations():
        max_x = max(form_o._coords, key=itemgetter(0))[0]
        max_y = max(form_o._coords, key=itemgetter(1))[1]
        for x in range(size[0]-max_x):
            for y in range(size[1]-max_y):
                if z[form_o, x, y].varValue > 0.999:
                    print(x, y)
                    print(form_o._coords)
                    print(form_o)
                    print("")

fig, ax = plt.subplots()
ax.add_patch(Rectangle((-0.55, -0.55), size[0]+0.55, size[1]+0.55, color='k', linewidth=0))
ax.add_patch(Rectangle((-0.5, -0.5), size[0], size[1], color='w', linewidth=0))
assignment = [[None for y in range(size[1])] for x in range(size[0])]
for form in forms:
    for form_o in form.get_all_orientations():
        max_x = max(form_o._coords, key=itemgetter(0))[0]
        max_y = max(form_o._coords, key=itemgetter(1))[1]
        for x in range(size[0]-max_x):
            for y in range(size[1]-max_y):
                if z[form_o, x, y].varValue > 0.999:
                    for xf, yf in form_o._coords:
                        assignment[x+xf][y+yf] = form_o
                    for i in range(len(form_o._coords)-1):
                        for j in range(i+1, len(form_o._coords)):
                            xi, yi = form_o._coords[i]
                            xj, yj = form_o._coords[j]
                            if abs(xi - xj) + abs(yi - yj) == 1:
                                width = 0.96
                                ax.add_patch(Rectangle((x+min(xi, xj)-width/2, y+min(yi, yj)-width/2), abs(xi-xj)+width, abs(yi-yj)+width, linewidth=0))

for x in range(size[0]-1):
    for y in range(size[1]-1):
        square = {assignment[x][y], assignment[x][y+1], assignment[x+1][y], assignment[x+1][y+1]}
        if len(square) == 1:
            ax.add_patch(Rectangle((x, y), 1, 1, linewidth=0))
for i in range(len(forbidden)-1):
    for j in range(i+1, len(forbidden)):
        if forbidden[i] not in date_to_coords(*date) and forbidden[j] not in date_to_coords(*date):
            xi, yi = forbidden[i]
            xj, yj = forbidden[j]
            if abs(xi - xj) + abs(yi - yj) == 1:
                width = 0.96
                ax.add_patch(Rectangle((min(xi, xj)-width/2, min(yi, yj)-width/2), abs(xi-xj)+width, abs(yi-yj)+width, linewidth=0, color='gray'))
ax.add_patch(Rectangle(tuple(map(sum, zip(date_to_coords(*date)[0], (-0.48, -0.48)))), 0.96, 0.96, color='k', linewidth=0))
ax.add_patch(Rectangle(tuple(map(sum, zip(date_to_coords(*date)[0], (-0.45, -0.45)))), 0.9, 0.9, color='w', linewidth=0))
ax.text(*date_to_coords(*date)[0], date[0], size='x-large', weight='bold', verticalalignment='center', horizontalalignment='center')
ax.add_patch(Rectangle(tuple(map(sum, zip(date_to_coords(*date)[1], (-0.48, -0.48)))), 0.96, 0.96, color='k', linewidth=0))
ax.add_patch(Rectangle(tuple(map(sum, zip(date_to_coords(*date)[1], (-0.45, -0.45)))), 0.9, 0.9, color='w', linewidth=0))
ax.text(*date_to_coords(*date)[1], month_to_text(date[1]), size='x-large', weight='bold', verticalalignment='center', horizontalalignment='center')
ax.set_xlim(-0.55, size[0]-0.45)
ax.set_ylim(-0.55, size[1]-0.45)
ax.set_aspect('equal')     
plt.axis('off')               
plt.show()

print("—".join(["+"] + ["—" for _ in range(size[0]-1)] + ["+"]))
for y in range(size[1]-1, -1, -1):
    if y < size[1]-1:
        output = "|"
        for x in range(size[0]):
            if assignment[x][y] == assignment[x][y+1]:
                if assignment[x][y] is None:
                    output += " "
                else:
                    output += "#"
            else:
                output += "—"
            if x < size[0]-1:
                # check the point between 4 coordinates
                square = {assignment[x][y], assignment[x][y+1], assignment[x+1][y], assignment[x+1][y+1]}
                above = {assignment[x][y+1], assignment[x+1][y+1]}
                below = {assignment[x][y], assignment[x+1][y]}
                left = {assignment[x][y], assignment[x][y+1]}
                right = {assignment[x+1][y], assignment[x+1][y+1]}
                if len(square) == 1:
                    # all 4 coordinates are the same object
                    if assignment[x][y] is None:
                        output += " "
                    else:
                        output += "#"
                else:
                    # check if above and below or left and right are one respective object
                    if len(above) == 1 and len(below) == 1:
                        output += "—"
                    elif len(left) == 1 and len(right) == 1:
                        output += "|"
                    else:
                        output += "+"
            else:
                output += "|"
        print(output)
    output = "|"
    for x in range(size[0]-1):
        if assignment[x][y] is None:
            output += " "
        else:
            output += "#"
        if assignment[x][y] == assignment[x+1][y]:
            if assignment[x][y] is None:
                output += " "
            else:
                output += "#"
        else:
            output += "|"
    if assignment[size[0]-1][y] is None:
        output += " |"
    else:
        output += "#|"
    print(output)
print("—".join(["+"] + ["—" for _ in range(size[0]-1)] + ["+"]))'''

