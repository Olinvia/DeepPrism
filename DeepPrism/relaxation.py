from gurobipy import *
import numpy as np
import torch


def f(x, y, tanh):
    # Calculates sigmoid(x) * tanh(y) or sigmoid(x) * y.
    if tanh:
        return np.tanh(y) / (1 + np.exp(-x))
    else:
        return y / (1 + np.exp(-x))


def sigma(x):
    # Calculates sigmoid(x).
    return 1 / (1 + np.exp(-x))


def ibp_bounds(lx, ux, ly, uy, tanh=True):
    # Calculates IBP bounds of [lx,ux] X [ly,uy].
    if type(lx) is torch.Tensor:
        lx = lx.item()
    if type(ux) is torch.Tensor:
        ux = ux.item()
    if type(ly) is torch.Tensor:
        ly = ly.item()
    if type(uy) is torch.Tensor:
        uy = uy.item()
    candidates = torch.Tensor(
        [f(lx, ly, tanh), f(lx, uy, tanh), f(ux, ly, tanh), f(ux, uy, tanh)]
    )
    return 0, 0, torch.min(candidates), 0, 0, torch.max(candidates)


def proper_roots(equ, lx, ux, ly, uy, var="x", fn=(lambda v: v)):
    # Equation solver and filter the roots satisfying the proper conditions.
    Xs = np.zeros([0])
    Ys = np.zeros([0])
    roots = np.roots(equ)
    if var == "x":
        for sx in roots:
            if np.iscomplex(sx) or sx >= 1 or sx <= 0:
                continue
            sx = np.real(sx)
            x = -np.log((1 - sx) / sx)
            y = fn(sx)
            if lx <= x and x <= ux and ly <= y and y <= uy:
                Xs = np.concatenate([Xs, [x]])
                Ys = np.concatenate([Ys, [y]])
    elif var == "y":
        for ty in roots:
            if np.iscomplex(ty) or ty >= 1 or ty <= -1:
                continue
            ty = np.real(ty)
            y = np.arctanh(ty)
            x = fn(ty)
            if lx <= x and x <= ux and ly <= y and y <= uy:
                Xs = np.concatenate([Xs, [x]])
                Ys = np.concatenate([Ys, [y]])
    return Xs, Ys


def get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh):
    # Cl calibration.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    if not tanh:
        # Case 2 only
        Xs1, Ys1 = (
            proper_roots([1, -1, Al / ly], lx, ux, ly, uy, var="x", fn=(lambda v: ly))
            if ly != 0
            else ([], [])
        )
        Xs2, Ys2 = (
            proper_roots([1, -1, Al / uy], lx, ux, ly, uy, var="x", fn=(lambda v: uy))
            if uy != 0
            else ([], [])
        )
        bndX = np.concatenate([bndX, Xs1, Xs2])
        bndY = np.concatenate([bndY, Ys1, Ys2])
    if tanh:
        # Case 1
        Xs1, Ys1 = proper_roots(
            [1, 0, Bl / sigma(lx) - 1], lx, ux, ly, uy, var="y", fn=(lambda v: lx)
        )
        Xs2, Ys2 = proper_roots(
            [1, 0, Bl / sigma(ux) - 1], lx, ux, ly, uy, var="y", fn=(lambda v: ux)
        )
        # Case 2
        Xs3, Ys3 = (
            proper_roots(
                [1, -1, Al / np.tanh(ly)], lx, ux, ly, uy, var="x", fn=(lambda v: ly)
            )
            if ly != 0
            else ([], [])
        )
        Xs4, Ys4 = (
            proper_roots(
                [1, -1, Al / np.tanh(uy)], lx, ux, ly, uy, var="x", fn=(lambda v: uy)
            )
            if uy != 0
            else ([], [])
        )
        # Case 3
        Xs5, Ys5 = proper_roots(
            [1, -2 - Bl, 1 + 2 * Bl, -Bl, -Al * Al],
            lx,
            ux,
            ly,
            uy,
            var="x",
            fn=(
                lambda v: np.arctanh(Al / v / (1 - v))
                if abs(Al / v / (1 - v)) < 1
                else uy + 1
            ),
        )
        bndX = np.concatenate([bndX, Xs1, Xs2, Xs3, Xs4, Xs5])
        bndY = np.concatenate([bndY, Ys1, Ys2, Ys3, Ys4, Ys5])

    delta = np.min(f(bndX, bndY, tanh) - Al * bndX - Bl * bndY - Cl)
    return delta


def get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh):
    # Cu calibration.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    if not tanh:
        # Case 2 only
        Xs1, Ys1 = (
            proper_roots([1, -1, Au / ly], lx, ux, ly, uy, var="x", fn=(lambda v: ly))
            if ly != 0
            else ([], [])
        )
        Xs2, Ys2 = (
            proper_roots([1, -1, Au / uy], lx, ux, ly, uy, var="x", fn=(lambda v: uy))
            if uy != 0
            else ([], [])
        )
        bndX = np.concatenate([bndX, Xs1, Xs2])
        bndY = np.concatenate([bndY, Ys1, Ys2])
    if tanh:
        # Case 1
        Xs1, Ys1 = proper_roots(
            [1, 0, Bu / sigma(lx) - 1], lx, ux, ly, uy, var="y", fn=(lambda v: lx)
        )
        Xs2, Ys2 = proper_roots(
            [1, 0, Bu / sigma(ux) - 1], lx, ux, ly, uy, var="y", fn=(lambda v: ux)
        )
        # Case 2
        Xs3, Ys3 = (
            proper_roots(
                [1, -1, Au / np.tanh(ly)], lx, ux, ly, uy, var="x", fn=(lambda v: ly)
            )
            if ly != 0
            else ([], [])
        )
        Xs4, Ys4 = (
            proper_roots(
                [1, -1, Au / np.tanh(uy)], lx, ux, ly, uy, var="x", fn=(lambda v: uy)
            )
            if uy != 0
            else ([], [])
        )
        # Case 3
        Xs5, Ys5 = proper_roots(
            [1, -2 - Bu, 1 + 2 * Bu, -Bu, -Au * Au],
            lx,
            ux,
            ly,
            uy,
            var="x",
            fn=(
                lambda v: np.arctanh(Au / v / (1 - v))
                if abs(Au / v / (1 - v)) < 1
                else uy + 1
            ),
        )
        bndX = np.concatenate([bndX, Xs1, Xs2, Xs3, Xs4, Xs5])
        bndY = np.concatenate([bndY, Ys1, Ys2, Ys3, Ys4, Ys5])

    delta = np.min(Au * bndX + Bu * bndY + Cu - f(bndX, bndY, tanh))
    return delta


def LB(lx, ux, ly, uy, tanh=True, n_samples=104, n_seed=242):
    # Calculate Al, Bl, Cl by sampling and linear programming.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    np.random.seed(n_seed)
    X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])
    Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])


    model = Model()
    model.setParam("OutputFlag", 0)

    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Al")
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bl")
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cl")

    model.addConstrs(
        (Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], tanh) for i in range(n_samples)),
        name="ctr",
    )

    obj = LinExpr()
    obj = np.sum(f(X, Y, tanh)) - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n_samples
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Al, Bl, Cl = model.getAttr("x", model.getVars())
        delta = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh)
        Cl += delta
        return Al, Bl, Cl
    else:
        return None, None, None


def UB(lx, ux, ly, uy, tanh=True, n_samples=104, n_seed=242):
    # Calculate Au, Bu, Cu by sampling and linear programming.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    np.random.seed(n_seed)
    X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])
    Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])


    model = Model()
    model.setParam("OutputFlag", 0)

    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Au")
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bu")
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cu")

    model.addConstrs(
        (Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], tanh) for i in range(n_samples)),
        name="ctr",
    )

    obj = LinExpr()
    obj = Au * np.sum(X) + Bu * np.sum(Y) + Cu * n_samples - np.sum(f(X, Y, tanh))
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Au, Bu, Cu = model.getAttr("x", model.getVars())
        delta = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh)
        Cu -= delta
        return Au, Bu, Cu
    else:
        return None, None, None


def bounds(lx, ux, ly, uy, tanh=True):
    # Caller function to obtain lower and upper bounding planes.
    if type(lx) is torch.Tensor:
        lx = lx.item()
    if type(ux) is torch.Tensor:
        ux = ux.item()
    if type(ly) is torch.Tensor:
        ly = ly.item()
    if type(uy) is torch.Tensor:
        uy = uy.item()
    Al, Bl, Cl = LB(lx, ux, ly, uy, tanh)
    Au, Bu, Cu = UB(lx, ux, ly, uy, tanh)
    return Al, Bl, Cl, Au, Bu, Cu


def LB_split(lx, ux, ly, uy, tanh=True, split_type=0, n_samples=200):
    # distance_1
    
    X,Y=np.meshgrid(np.linspace(lx,ux,10),np.linspace(ly,uy,20))
    X=X.reshape(-1)
    Y=Y.reshape(-1)

    if split_type == 11:
        sel = (ux - lx) * (Y - ly) <= (uy - ly) * (X - lx)
    elif split_type == 12:
        sel = (ux - lx) * (Y - ly) >= (uy - ly) * (X - lx)
    elif split_type == 21:
        sel = (ux - lx) * (Y - ly) <= (ly - uy) * (X - ux)
    elif split_type == 22:
        sel = (ux - lx) * (Y - ly) >= (ly - uy) * (X - ux)
    X, Y = X[sel], Y[sel]
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, X])
    Y = np.concatenate([bndY, Y])
    n = X.shape[0]

    model = Model()
    model.setParam("OutputFlag", 0)

    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Al")
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bl")
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cl")

    model.addConstrs(
        (Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], tanh) for i in range(n)),
        name="ctr",
    )

    obj = LinExpr()
    obj = np.sum(f(X, Y, tanh)) - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n
    if split_type == 11:
        obj -= f(lx, uy, tanh) - Al * lx - Bl * uy - Cl
    elif split_type == 12:
        obj -= f(ux, ly, tanh) - Al * ux - Bl * ly - Cl
    elif split_type == 21:
        obj -= f(ux, uy, tanh) - Al * ux - Bl * uy - Cl
    elif split_type == 22:
        obj -= f(lx, ly, tanh) - Al * lx - Bl * ly - Cl
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Al, Bl, Cl = model.getAttr("x", model.getVars())
        delta = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh)
        Cl += delta
        return Al, Bl, Cl, model.objVal / (n - 1)
    else:
        return None, None, None, None


def UB_split(lx, ux, ly, uy, tanh=True, split_type=0, n_samples=200):
    # distance_2
    X,Y=np.meshgrid(np.linspace(lx,ux,10),np.linspace(ly,uy,20))
    X=X.reshape(-1)
    Y=Y.reshape(-1)

    if split_type == 11:
        sel = (ux - lx) * (Y - ly) <= (uy - ly) * (X - lx)
    elif split_type == 12:
        sel = (ux - lx) * (Y - ly) >= (uy - ly) * (X - lx)
    elif split_type == 21:
        sel = (ux - lx) * (Y - ly) <= (ly - uy) * (X - ux)
    elif split_type == 22:
        sel = (ux - lx) * (Y - ly) >= (ly - uy) * (X - ux)
    X, Y = X[sel], Y[sel]
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, X])
    Y = np.concatenate([bndY, Y])
    n = X.shape[0]

    model = Model()
    model.setParam("OutputFlag", 0)

    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Au")
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bu")
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cu")

    model.addConstrs(
        (Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], tanh) for i in range(n)),
        name="ctr",
    )

    obj = LinExpr()
    obj = Au * np.sum(X) + Bu * np.sum(Y) + Cu * n - np.sum(f(X, Y, tanh))
    if split_type == 11:
        obj += f(lx, uy, tanh) - Au * lx - Bu * uy - Cu
    elif split_type == 12:
        obj += f(ux, ly, tanh) - Au * ux - Bu * ly - Cu
    elif split_type == 21:
        obj += f(ux, uy, tanh) - Au * ux - Bu * uy - Cu
    elif split_type == 22:
        obj += f(lx, ly, tanh) - Au * lx - Bu * ly - Cu
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Au, Bu, Cu = model.getAttr("x", model.getVars())
        delta = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh)
        Cu -= delta
        return Au, Bu, Cu, model.objVal / (n - 1)
    else:
        return None, None, None, None


def L2UB(lx, ux, ly, uy, tanh=True, n_samples=100):
    # volume
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])
    Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])

    # 创建gurobipy优化模型对象
    model = Model()
    # 0: 禁止输出任何信息。
    # 1: 输出关键信息，比如最优解的值、求解时间等。
    # 2: 输出更详细的信息，包括每个步骤的详细信息和调试信息。
    # 3: 输出更加详细的信息，包括每个步骤的更加详细的信息和调试信息。
    model.setParam("OutputFlag", 0)

    # 添加变量 范围不限
    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Al")
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bl")
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cl")
    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Au")
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bu")
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cu")

    model.addConstrs(
        (Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], tanh) for i in range(n_samples)),
        name="ctrl",
    )

    model.addConstrs(
        (Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], tanh) for i in range(n_samples)),
        name="ctru",
    )

    obj = LinExpr()
    mid_x = (lx + ux) / 2
    mid_y = (ly + uy) / 2
    height = Au * mid_x + Bu * mid_y + Cu - Al * mid_x - Bl * mid_y - Cl
    obj = (ux - lx) * (uy - ly) * height
    model.setObjective(obj, GRB.MINIMIZE)       # 最小化距离

    model.optimize()                            # 运行优化器

    if model.status == GRB.Status.OPTIMAL:      # 如果最优了
        Al, Bl, Cl, Au, Bu, Cu = model.getAttr("x", model.getVars())
        delta_Cl = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh)
        Cl += delta_Cl
        delta_Cu = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh)
        Cu -= delta_Cu
        return Al, Bl, Cl, Au, Bu, Cu
    else:
        return None, None, None, None, None, None


def L2UBZ(lx, ux, ly, uy, tanh=True, n_samples=104, n_seed=42):
    # volume + area
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])
    Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])

    model = Model()
    model.setParam("OutputFlag", 0)

    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Al")
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bl")
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cl")
    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Au")
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bu")
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cu")
    Zl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Zl")
    Zu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Zu")
    y1 = model.addVar(vtype=GRB.BINARY, name="y1")
    y2 = model.addVar(vtype=GRB.BINARY, name="y2")
    y3 = model.addVar(vtype=GRB.BINARY, name="y3")
    y4 = model.addVar(vtype=GRB.BINARY, name="y4")
    y5 = model.addVar(vtype=GRB.BINARY, name="y5")
    y6 = model.addVar(vtype=GRB.BINARY, name="y6")
    y7 = model.addVar(vtype=GRB.BINARY, name="y7")
    y8 = model.addVar(vtype=GRB.BINARY, name="y8")

    model.addConstrs(
        (Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], tanh) for i in range(n_samples)),
        name="ctrl",
    )

    model.addConstrs(
        (Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], tanh) for i in range(n_samples)),
        name="ctru",
    )

    model.addConstr(Zl <= Al * lx + Bl * ly + Cl, name="al")
    model.addConstr(Zl <= Al * lx + Bl * uy + Cl, name="bl")
    model.addConstr(Zl <= Al * ux + Bl * ly + Cl, name="cl")
    model.addConstr(Zl <= Al * ux + Bl * uy + Cl, name="dl")

    model.addConstr(Zl == (Al * lx + Bl * ly + Cl) * y1 + (Al * lx + Bl * uy + Cl) * y2
                    + (Al * ux + Bl * ly + Cl) * y3 + (Al * ux + Bl * uy + Cl) * y4, name="el")

    model.addConstr(y1 + y2 + y3 + y4 == 1, name="fl")

    model.addConstr(Zu >= Au * lx + Bu * ly + Cu, name="au")
    model.addConstr(Zu >= Au * lx + Bu * uy + Cu, name="bu")
    model.addConstr(Zu >= Au * ux + Bu * ly + Cu, name="cu")
    model.addConstr(Zu >= Au * ux + Bu * uy + Cu, name="du")

    model.addConstr(Zu == (Au * lx + Bu * ly + Cu) * y5 + (Au * lx + Bu * uy + Cu) * y6
                    + (Au * ux + Bu * ly + Cu) * y7 + (Au * ux + Bu * uy + Cu) * y8, name="eu")

    model.addConstr(y5 + y6 + y7 + y8 == 1, name="fu")

    obj = LinExpr()
    #obj1 = LinExpr()
    #obj1 = (np.sum(f(X, Y, tanh)) - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n_samples) / n_samples
    #obj2 = LinExpr()
    #obj2 = (Au * np.sum(X) + Bu * np.sum(Y) + Cu * n_samples - np.sum(f(X, Y, tanh))) / n_samples

    # with open('num.txt', 'r') as file:
    #    content = file.read().strip()
    # alpha = float(content)
    # obj = alpha* height + (1-alpha)*(Zu - Zl)

    mid_x = (lx + ux) / 2
    mid_y = (ly + uy) / 2
    height = Au * mid_x + Bu * mid_y + Cu - Al * mid_x - Bl * mid_y - Cl
    # distance = (Au * np.sum(X) + Bu * np.sum(Y) + Cu * n_samples
    #            - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n_samples) / n_samples
    obj = 0.674 * height + 0.326 * (Zu - Zl)
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        optimal_values = model.getAttr("x", model.getVars())
        Al = optimal_values[0]
        Bl = optimal_values[1]
        Cl = optimal_values[2]
        Au = optimal_values[3]
        Bu = optimal_values[4]
        Cu = optimal_values[5]
        delta_Cl = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh)
        Cl += delta_Cl
        delta_Cu = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh)
        Cu -= delta_Cu

        return Al, Bl, Cl, Au, Bu, Cu
    else:
        return None, None, None, None, None, None

def LUB_split(lx, ux, ly, uy, tanh=True, split_type=0, n_samples=200):
    # Get lower bound plane with triangular domain.
    X = np.random.uniform(lx, ux, n_samples)
    Y = np.random.uniform(ly, uy, n_samples)

    if split_type == 11:
        sel = (ux - lx) * (Y - ly) <= (uy - ly) * (X - lx)
    elif split_type == 12:
        sel = (ux - lx) * (Y - ly) >= (uy - ly) * (X - lx)
    elif split_type == 21:
        sel = (ux - lx) * (Y - ly) <= (ly - uy) * (X - ux)
    elif split_type == 22:
        sel = (ux - lx) * (Y - ly) >= (ly - uy) * (X - ux)
    X, Y = X[sel], Y[sel]
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, X])
    Y = np.concatenate([bndY, Y])
    n = X.shape[0]

    model = Model()
    model.setParam("OutputFlag", 0)

    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Al")
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bl")
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cl")
    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Au")
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bu")
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cu")
    Zl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Zl")
    Zu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Zu")
    y1 = model.addVar(vtype=GRB.BINARY, name="y1")
    y2 = model.addVar(vtype=GRB.BINARY, name="y2")
    y3 = model.addVar(vtype=GRB.BINARY, name="y3")
    y4 = model.addVar(vtype=GRB.BINARY, name="y4")
    y5 = model.addVar(vtype=GRB.BINARY, name="y5")
    y6 = model.addVar(vtype=GRB.BINARY, name="y6")
    y7 = model.addVar(vtype=GRB.BINARY, name="y7")
    y8 = model.addVar(vtype=GRB.BINARY, name="y8")

    model.addConstrs(
        (Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], tanh) for i in range(n)),
        name="ctrl",
    )

    model.addConstrs(
        (Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], tanh) for i in range(n)),
        name="ctru",
    )
    

    obj = LinExpr()
    # (lx,ly)   (ux,ly)     (ux,uy)
    if split_type == 11:
        model.addConstr(Zl <= Al * lx + Bl * ly + Cl,name="al")
        model.addConstr(Zl <= Al * ux + Bl * ly + Cl, name="cl")
        model.addConstr(Zl <= Al * ux + Bl * uy + Cl, name="dl")
        model.addConstr(Zl == (Al * lx + Bl * ly + Cl)*y1 + (Al * ux + Bl * ly + Cl)*y3 + (Al * ux + Bl * uy + Cl)*y4, name="el")
        model.addConstr(y1 + y3 + y4 == 1,name="fl")
    
        model.addConstr(Zu >= Au * lx + Bu * ly + Cu, name="au")
        model.addConstr(Zu >= Au * ux + Bu * ly + Cu, name="cu")
        model.addConstr(Zu >= Au * ux + Bu * uy + Cu, name="du")
        model.addConstr(Zu == (Au * lx + Bu * ly + Cu) * y5 + (Au * ux + Bu * ly + Cu) * y7 + (Au * ux + Bu * uy + Cu) * y8, name="eu")
        model.addConstr(y5 + y7 + y8 == 1, name="fu")
    
        zu = ((Au * lx + Bu * ly + Cu) + (Au * ux + Bu * ly + Cu) + (Au * ux + Bu * uy + Cu)) / 3
        zl = ((Al * lx + Bl * ly + Cl) + (Al * ux + Bl * ly + Cl) + (Al * ux + Bl * uy + Cl)) / 3
        height = zu - zl
        obj = 0.674 * height + 0.326 * (Zu - Zl)
    # (lx,ly)   (lx,uy)     (ux,uy)
    elif split_type == 12:
        model.addConstr(Zl <= Al * lx + Bl * ly + Cl,name="al")
        model.addConstr(Zl <= Al * lx + Bl * uy + Cl, name="bl")
        model.addConstr(Zl <= Al * ux + Bl * uy + Cl, name="dl")
        model.addConstr(Zl == (Al * lx + Bl * ly + Cl)*y1 + (Al * lx + Bl * uy + Cl)*y2 + (Al * ux + Bl * uy + Cl)*y4, name="el")
    
        model.addConstr(y1 + y2 + y4 == 1,name="fl")
    
        model.addConstr(Zu >= Au * lx + Bu * ly + Cu, name="au")
        model.addConstr(Zu >= Au * lx + Bu * uy + Cu, name="bu")
        model.addConstr(Zu >= Au * ux + Bu * uy + Cu, name="du")
    
        model.addConstr(Zu == (Au * lx + Bu * ly + Cu) * y5 + (Au * lx + Bu * uy + Cu) * y6
                        + (Au * ux + Bu * ly + Cu) * y7 + (Au * ux + Bu * uy + Cu) * y8, name="eu")
    
        model.addConstr(y5 + y6 + y8 == 1, name="fu")
        
        zu = ((Au * lx + Bu * ly + Cu) + (Au * lx + Bu * uy + Cu) + (Au * ux + Bu * uy + Cu)) / 3
        zl = ((Al * lx + Bl * ly + Cl) + (Al * lx + Bl * uy + Cl) + (Al * ux + Bl * uy + Cl)) / 3
        height = zu - zl
        obj = 0.674 * height + 0.326 * (Zu - Zl)
    # (lx,ly)   (lx,uy)     (ux,ly)
    elif split_type == 21:
        model.addConstr(Zl <= Al * lx + Bl * ly + Cl,name="al")
        model.addConstr(Zl <= Al * lx + Bl * uy + Cl, name="bl")
        model.addConstr(Zl <= Al * ux + Bl * ly + Cl, name="cl")
        model.addConstr(Zl == (Al * lx + Bl * ly + Cl)*y1 + (Al * lx + Bl * uy + Cl)*y2 + (Al * ux + Bl * ly + Cl)*y3, name="el")
        model.addConstr(y1 + y2 + y3 == 1,name="fl")
    
        model.addConstr(Zu >= Au * lx + Bu * ly + Cu, name="au")
        model.addConstr(Zu >= Au * lx + Bu * uy + Cu, name="bu")
        model.addConstr(Zu >= Au * ux + Bu * ly + Cu, name="cu")
        model.addConstr(Zu == (Au * lx + Bu * ly + Cu) * y5 + (Au * lx + Bu * uy + Cu) * y6 + (Au * ux + Bu * ly + Cu) * y7, name="eu")
        model.addConstr(y5 + y6 + y7 == 1, name="fu")
    
        zu = ((Au * lx + Bu * ly + Cu) + (Au * lx + Bu * uy + Cu) + (Au * ux + Bu * ly + Cu)) / 3
        zl = ((Al * lx + Bl * ly + Cl) + (Al * lx + Bl * uy + Cl) + (Al * ux + Bl * ly + Cl)) / 3
        height = zu - zl
        obj = 0.674 * height + 0.326 * (Zu - Zl)
    # (lx,uy)   (ux,ly)     (ux,uy)
    elif split_type == 22:
        model.addConstr(Zl <= Al * lx + Bl * uy + Cl, name="bl")
        model.addConstr(Zl <= Al * ux + Bl * ly + Cl, name="cl")
        model.addConstr(Zl <= Al * ux + Bl * uy + Cl, name="dl")
        model.addConstr(Zl == (Al * lx + Bl * uy + Cl)*y2 + (Al * ux + Bl * ly + Cl)*y3 + (Al * ux + Bl * uy + Cl)*y4, name="el")
        model.addConstr(y2 + y3 + y4 == 1,name="fl")
    
        model.addConstr(Zu >= Au * lx + Bu * uy + Cu, name="bu")
        model.addConstr(Zu >= Au * ux + Bu * ly + Cu, name="cu")
        model.addConstr(Zu >= Au * ux + Bu * uy + Cu, name="du")
        model.addConstr(Zu == (Au * lx + Bu * uy + Cu) * y6 + (Au * ux + Bu * ly + Cu) * y7 + (Au * ux + Bu * uy + Cu) * y8, name="eu")
        model.addConstr(y6 + y7 + y8 == 1, name="fu")
    
        zu = ((Au * lx + Bu * uy + Cu) + (Au * ux + Bu * ly + Cu) + (Au * ux + Bu * ly + Cu)) / 3
        zl = ((Al * lx + Bl * uy + Cl) + (Al * ux + Bl * ly + Cl) + (Al * ux + Bl * ly + Cl)) / 3
        height = zu - zl
        obj = 0.674 * height + 0.326 * (Zu - Zl)
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        optimal_values = model.getAttr("x", model.getVars())
        Al = optimal_values[0]
        Bl = optimal_values[1]
        Cl = optimal_values[2]
        Au = optimal_values[3]
        Bu = optimal_values[4]
        Cu = optimal_values[5]
        delta_Cl = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh)
        Cl += delta_Cl
        delta_Cu = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh)
        Cu -= delta_Cu
        return Al, Bl, Cl, Au, Bu, Cu
    else:
        return None, None, None, None, None, None

if __name__ == "__main__":
    print(LB(-1, 2, -2, 3))
    print(UB(-1, 2, -2, 3))
