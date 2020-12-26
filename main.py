def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def solveSudoku(board):
    """
    :type board: List[List[str]]
    :rtype: None Do not return anything, modify board in-place instead.
    """
    import numpy as np

    V = np.zeros(81, dtype=int).reshape((9, 9))
    Vx = [[set({1, 2, 3, 4, 5, 6, 7, 8, 9}) for _ in range(9)] for __ in range(9)]

    vals_to_set = set()

    for i in range(9):
        for j in range(9):
            cchar = board[i][j]
            if (cchar != "."):
                newvals, flagc = writeValueInCell(V, Vx, i, j, int(cchar))
                vals_to_set.update(newvals)

    propagateValues(V, Vx, vals_to_set)

    values_to_try = allValuesToTry(Vx)

    V_copy, Vx_copy, remaining_empty, contradiction = trySettingValuesDepth1(V, Vx, values_to_try)

    remaining_empty = remainingConstraints(Vx)
    if (remaining_empty > 0):
        values_to_try = allValuesToTry(Vx)
        V_copy, Vx_copy, remaining_empty, contradiction = trySettingValuesDepth2(V, Vx, values_to_try)
        if (remaining_empty == 0):
            V = V_copy

    print(V)
    print("Checking valid: ", checkValid(V))

    for i in range(9):
        for j in range(9):
            board[i][j] = str(V[i][j])


def trySettingValuesDepth2(V, Vx, values_to_try):
    V_copy  = V      # These two assignments are done to have V_copy and Vx_copy as global
    Vx_copy = Vx
    contradiction = False

    print("Depth 2 values to try: ", values_to_try)

    while (len(values_to_try) > 0):
        val_to_try = values_to_try.pop()
        i_x, j_x, k_x = unpackValue(val_to_try)

        if (not Vx[i_x][j_x].issuperset({k_x})):
#            print("Value ", i_x, j_x, k_x, " not allowed")
            continue

        if (V[i_x][j_x] != 0 and V[i_x][j_x] != k_x):
            continue

        V_copy, Vx_copy = copyVVx(V, Vx)

        remaining_constraints, contradiction = hardSet(V_copy, Vx_copy, i_x, j_x, k_x)

        if (contradiction):
            continue

        currvals_to_try = allValuesToTry(Vx_copy)

        V_copy2, Vx_copy2, remaining_empty, contradiction = trySettingValuesDepth1(V_copy, Vx_copy, currvals_to_try)

        if (contradiction == True):
            remaining_empty, contradiction = eliminateValue(V, Vx, i_x, j_x, k_x)
            if (remaining_empty == 0):
                print("Solved!!!")
                return V, Vx, remaining_empty, contradiction
        if (remaining_empty == 0):
            print("Solved!!")
            return V_copy2, Vx_copy2, remaining_empty, contradiction

    return V, Vx, remainingConstraints(Vx), contradiction

def trySettingValuesDepth1(V, Vx, values_to_try):
    V_copy  = V      # These two assignments are done to have V_copy and Vx_copy as global
    Vx_copy = Vx
    contradiction = False

    while (len(values_to_try) > 0):
        val_to_try = values_to_try.pop()
        i_x, j_x, k_x = unpackValue(val_to_try)

        if (not Vx[i_x][j_x].issuperset({k_x})):
#            print("Value ", i_x, j_x, k_x, " not allowed")
            continue

        V_copy, Vx_copy, remaining_empty, contradiction = trySetting(V, Vx, i_x, j_x, k_x)

        if (contradiction == True):
#            print("Value ", i_x, j_x, k_x, " is invalid, will eliminate it")
            remaining_empty, contradiction = eliminateValue(V, Vx, i_x, j_x, k_x)
            if (contradiction == True):
                return V_copy, Vx_copy, remaining_empty, contradiction
            if (remaining_empty == 0):
                print("Solved now! ")
                return V, Vx, remaining_empty, contradiction
        if (remaining_empty == 0):
            print("> Solved!")
            return V_copy, Vx_copy, remaining_empty, contradiction

    return V_copy, Vx_copy, remainingConstraints(Vx_copy), contradiction


def unpackValue(val):
    i_x, j_x, k_x = (val % 81) // 9, val % 9, val // 81
    return i_x, j_x, k_x

def propagateValues(V, Vx, vals_to_set):

    # returns the number of remaining empty cells, and a flag that is true if conflict was detected

    while (len(vals_to_set) > 0):
        val_to_set = vals_to_set.pop()
        i_x, j_x = val_to_set // 9, val_to_set % 9
        k_x = next(iter(Vx[i_x][j_x]))
        newvals, flagc = writeValueInCell(V, Vx, i_x, j_x, k_x)
        if (flagc == True):
            return remainingConstraints(Vx), True
        vals_to_set.update( newvals )

    return remainingConstraints(Vx), False

def eliminateValue(V, Vx, i_x, j_x, k_x):
    if (Vx[i_x][j_x].issuperset({k_x})):
        Vx[i_x][j_x].difference_update({k_x})
        if (len(Vx[i_x][j_x]) == 1):
            val = next(iter(Vx[i_x][j_x]))
            rc, currflag = hardSet(V, Vx, i_x, j_x, val)
            return rc, currflag
    return remainingConstraints(Vx), False


def copyVVx(V, Vx):
    import numpy as np

    V_copy = np.zeros(81, dtype=int).reshape((9, 9))
    Vx_copy = [[set() for _ in range(9)] for __ in range(9)]
    for i in range(9):
        for j in range(9):
            V_copy[i][j] = V[i][j]
            Vx_copy[i][j] = Vx[i][j].copy()
    return V_copy, Vx_copy


def trySetting(V, Vx, i_x, j_x, val_to_try):
    if (V[i_x][j_x] != 0):
        if (V[i_x][j_x] == val_to_try):
            return V, Vx, remainingConstraints(Vx), False
        else:
            return V, Vx, remainingConstraints(Vx), True
    V_copy, Vx_copy = copyVVx(V, Vx)
    valset = {val_to_try}
    assert valset.issubset(Vx[i_x][j_x])

    rc, currflag = hardSet(V_copy, Vx_copy, i_x, j_x, val_to_try)
    return V_copy, Vx_copy, rc, currflag


def hardSet(V, Vx, i_x, j_x, val):

    vals_to_set, flag_conflict = writeValueInCell(V, Vx, i_x, j_x, val)

    if (flag_conflict):
        return remainingConstraints(Vx), flag_conflict

#    print("Ending V: ", V)
    return propagateValues(V, Vx, vals_to_set)


def exactlyNcases(Vx, N):
    ncases = set()
    for i in range(9):
        for j in range(9):
            if (len(Vx[i][j]) == N):
                for k in Vx[i][j]:
                    ncases.add(81*k + 9 * i + j)
    return ncases


def allValuesToTry(Vx):
    all_values = set()
    for i in range(2, 10):
        all_values.update(exactlyNcases(Vx, i))
    return all_values


def otherCellsInSquare(i, j):
    other_cells = list()
    for i_x in range(3 * (i//3), 3 * (i//3) +3):
        for j_x in range(3 * (j//3), 3 * (j//3) +3):
            if (i_x != i or j_x != j):
                other_cells.append([i_x, j_x])
    return other_cells


def remainingConstraints(Vx):
    rval = 0
    for i in range(9):
        for j in range(9):
            if (len(Vx[i][j]) > 1):
                rval = rval + 1
    return rval


def writeValueInCell(V, Vx, i, j, k):
    valsToSet = set()  # Contains the coordinates of additional cells that are constrained to a specific value
    #                    Coordinates are encoded in 0-80, and the value can be retrieved from Vx

    flag_conflict = False
    if (V[i][j] != k and V[i][j] != 0):
        assert(len(Vx[i][j])==1 and next(iter(Vx[i][j])) == V[i][j])
        flag_conflict = True
        return valsToSet, flag_conflict

    V[i][j] = k
    Vx[i][j] = {k}
    templen = 0

    for l in range(9):

        if (l != j):
            templen = len(Vx[i][l])
            Vx[i][l].discard(k)
            if (len(Vx[i][l]) == 0):
 #               print("Conflict 1")
                flag_conflict = True
            elif (len(Vx[i][l]) == 1 and templen > 1):
                valsToSet.add(9*i +l)

        if (l != i):
            templen = len(Vx[l][j])
            Vx[l][j].discard(k)
            if (len(Vx[l][j]) == 0):
#                print("Conflict 2", l, j, V[l][j], Vx[l][j])
                flag_conflict = True
            elif (len(Vx[l][j]) == 1 and templen > 1):
                valsToSet.add(9*l +j)


    other_cells = otherCellsInSquare(i, j)
    for c in other_cells:
        templen = len(Vx[c[0]][c[1]])
        Vx[c[0]][c[1]].discard(k)
        if (len(Vx[c[0]][c[1]]) == 0):
            flag_conflict = True
        elif (len(Vx[c[0]][c[1]]) == 1 and templen > 1):
            valsToSet.add(9*c[0] + c[1])

    return valsToSet, flag_conflict


def checkValid(V):
    for i in range(9):
        for k in range(1, 10):
            cnt = 0
            for j in range(9):
                if (V[i][j] == k):
                    cnt = cnt +1
            if (cnt > 1):
                print ("Row ", i)
                return False
            cnt = 0
            for j in range(9):
                if (V[j][i] == k):
                    cnt = cnt + 1
            if (cnt > 1):
                print("Column ", i)
                return False
            cnt = 0
            for ix in range(3):
                for jx in range(3):
                    if (V[3*(i//3) + ix][3*(i%3) + jx] == k):
                        cnt = cnt + 1
            if (cnt > 1):
                print("Square ", i, k, i//3, i%3)
                return False
    return True


def coordsToNum(i, j):
    return int(3 * (i // 3)) + j // 3


def printVx(Vx):
    for i in range(9):
        print(Vx[i])


if __name__ == '__main__':
    print_hi('PyCharm')

    solveSudoku([["1",".",".",".","7",".",".","3","."],
                 ["8","3",".","6",".",".",".",".","."],
                 [".",".","2","9",".",".","6",".","8"],
                 ["6",".",".",".",".","4","9",".","7"],
                 [".","9",".",".",".",".",".","5","."],
                 ["3",".","7","5",".",".",".",".","4"],
                 ["2",".","3",".",".","9","1",".","."],
                 [".",".",".",".",".","2",".","4","3"],
                 [".","4",".",".","8",".",".",".","9"]])

    solveSudoku([["5","3",".",".","7",".",".",".","."],
                 ["6",".",".","1","9","5",".",".","."],
                 [".","9","8",".",".",".",".","6","."],
                 ["8",".",".",".","6",".",".",".","3"],
                 ["4",".",".","8",".","3",".",".","1"],
                 ["7",".",".",".","2",".",".",".","6"],
                 [".","6",".",".",".",".","2","8","."],
                 [".",".",".","4","1","9",".",".","5"],
                 [".",".",".",".","8",".",".","7","9"]])

    solveSudoku([[".",".",".","2",".",".",".","6","3"],
                 ["3",".",".",".",".","5","4",".","1"],
                 [".",".","1",".",".","3","9","8","."],
                 [".",".",".",".",".",".",".","9","."],
                 [".",".",".","5","3","8",".",".","."],
                 [".","3",".",".",".",".",".",".","."],
                 [".","2","6","3",".",".","5",".","."],
                 ["5",".","3","7",".",".",".",".","8"],
                 ["4","7",".",".",".","1",".",".","."]])

    solveSudoku([[".",".",".","7",".","9","3",".","4"],
                 [".",".",".",".",".",".",".","6","9"],
                 [".",".",".",".","4",".",".",".","."],
                 [".","9",".",".","8","7","2",".","5"],
                 [".","3",".","2",".","5",".","4","."],
                 ["5",".","7","4","6",".",".","8","."],
                 [".",".",".",".","7",".",".",".","."],
                 ["1","4",".",".",".",".",".",".","."],
                 ["2",".","3","8",".","1",".",".","."]])

    solveSudoku([["8",".",".",".","4",".","9","1","."],
                 [".",".","3","5",".",".",".",".","."],
                 [".",".",".",".",".",".","2",".","."],
                 ["5","8","6",".",".",".",".",".","."],
                 [".",".",".",".",".","9",".",".","."],
                 [".",".",".",".","7",".",".",".","8"],
                 [".",".",".",".","3",".","7",".","6"],
                 [".",".","9",".",".",".",".",".","5"],
                 ["3","7",".",".","2",".",".",".","."]])

    solveSudoku([[".",".","6",".",".",".",".","7","."],
                ["8",".",".",".",".",".",".","4","6"],
                 [".",".","5","3","9",".",".",".","."],
                 [".",".",".",".",".",".","3",".","1"],
                 ["2",".","7",".",".",".",".","6","9"],
                 [".",".",".","5",".",".",".",".","."],
                 [".",".",".",".",".",".",".",".","."],
                 [".",".","9","7",".",".",".","1","5"],
                 [".","6",".",".","2","8",".",".","."]])

# Hard sudoku
    solveSudoku([["1",".",".",".",".","7",".","9","."],
                 [".","3",".",".","2",".",".",".","8"],
                 [".",".","9","6",".",".","5",".","."],
                 [".",".","5","3",".",".","9",".","."],
                 [".","1",".",".","8",".",".",".","2"],
                 ["6",".",".",".",".","4",".",".","."],
                 ["3",".",".",".",".",".",".","1","."],
                 [".","4",".",".",".",".",".",".","7"],
                 [".",".","7",".",".",".","3",".","."]])

# "World's toughest sudoku is here, can you crack it?"
    solveSudoku([[".",".","5","3",".",".",".",".","."],
                 ["8",".",".",".",".",".",".","2","."],
                 [".","7",".",".","1",".","5",".","."],
                 ["4",".",".",".",".","5","3",".","."],
                 [".","1",".",".","7",".",".",".","6"],
                 [".",".","3","2",".",".",".","8","."],
                 [".","6",".","5",".",".",".",".","9"],
                 [".",".","4",".",".",".",".","3","."],
                 [".",".",".",".",".","9","7",".","."]])

