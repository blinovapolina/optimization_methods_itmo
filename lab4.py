INF = float('-inf')


class BrigadeDistributor:
    def __init__(self, n, f1, f2, f3, d2=(0, 2, 4), min_x3=1):
        self.n = n
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.d2 = d2
        self.min_x3 = min_x3

        self.F1 = [INF] * (n + 1)
        self.F2 = [INF] * (n + 1)
        self.F3 = INF
        self.X1 = [0] * (n + 1)
        self.X2 = [0] * (n + 1)
        self.X3 = 0

    def _stage1(self):
        for s in range(self.n + 1):
            self.F1[s] = self.f1[s]
            self.X1[s] = s

    def _stage2(self):
        for s in range(self.n + 1):
            best = INF
            best_x2 = 0
            for x2 in self.d2:
                if x2 <= s:
                    val = self.f2[x2] + self.F1[s - x2]
                    if val > best:
                        best = val
                        best_x2 = x2
            self.F2[s] = best
            self.X2[s] = best_x2

    def _stage3(self):
        best = INF
        best_x3 = 0
        for x3 in range(self.min_x3, self.n + 1):
            val = self.f3[x3] + self.F2[self.n - x3]
            if val > best:
                best = val
                best_x3 = x3
        self.F3 = best
        self.X3 = best_x3

    def solve(self):
        self._stage1()
        self._stage2()
        self._stage3()

        x3 = self.X3
        x2 = self.X2[self.n - x3]
        x1 = self.n - x3 - x2
        return x1, x2, x3, self.F3


if __name__ == "__main__":
    n = 6

    f1 = [0, 4, 7, 11, 14, 16, 17]
    f2 = [0, 3, 8, 12, 14, 15, 16]
    f3 = [0, 5, 8, 10, 13, 16, 18]

    solver = BrigadeDistributor(n, f1, f2, f3, d2=(0, 2, 4), min_x3=1)
    x1, x2, x3, total = solver.solve()

    print("Оптимальное распределение бригад:")
    print(f"Мост: {x1} бригад -> {f1[x1]} операций")
    print(f"Тоннель: {x2} бригад -> {f2[x2]} операций")
    print(f"Развязка: {x3} бригад -> {f3[x3]} операций")
    print(f"Всего бригад: {x1 + x2 + x3}")
    print(f"Максимум операций: {total}")