import heapq
import copy
import time

# Estado objetivo
goal_state = [[1, 2, 3],
              [8, 0, 4],
              [7, 6, 5]]

# ---------------- Funções de apoio ----------------

def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def state_to_tuple(state):
    return tuple(item for row in state for item in row)

def print_board(state):
    for row in state:
        print(" ".join(str(x) if x != 0 else "_" for x in row))
    print()

def get_neighbors(state):
    neighbors = []
    x, y = find_zero(state)
    moves = [(-1,0), (1,0), (0,-1), (0,1)]  # cima, baixo, esquerda, direita

    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = copy.deepcopy(state)
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(new_state)
    return neighbors

# ---------------- Heurísticas ----------------

# h1 = número de peças fora do lugar
def h_misplaced(state):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                count += 1
    return count

# h2 = distância de Manhattan
def h_manhattan(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                goal_x, goal_y = divmod(value - 1, 3)
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

# ---------------- Algoritmo A* ----------------

def a_star(start_state, heuristic='manhattan'):
    open_list = []
    heapq.heappush(open_list, (0, start_state, []))
    visited = set()
    visited.add(state_to_tuple(start_state))

    while open_list:
        cost, current_state, path = heapq.heappop(open_list)

        if current_state == goal_state:
            return path + [current_state]

        g = len(path)

        for neighbor in get_neighbors(current_state):
            t = state_to_tuple(neighbor)
            if t not in visited:
                visited.add(t)

                # Define a heurística usada
                if heuristic == 'manhattan':
                    h = h_manhattan(neighbor)
                else:
                    h = h_misplaced(neighbor)

                f = g + 1 + h
                heapq.heappush(open_list, (f, neighbor, path + [current_state]))
    return None

# ---------------- Movimentação manual ----------------

def move(state, direction):
    x, y = find_zero(state)
    new_state = copy.deepcopy(state)

    if direction == 'w' and x > 0:
        new_state[x][y], new_state[x-1][y] = new_state[x-1][y], new_state[x][y]
    elif direction == 's' and x < 2:
        new_state[x][y], new_state[x+1][y] = new_state[x+1][y], new_state[x][y]
    elif direction == 'a' and y > 0:
        new_state[x][y], new_state[x][y-1] = new_state[x][y-1], new_state[x][y]
    elif direction == 'd' and y < 2:
        new_state[x][y], new_state[x][y+1] = new_state[x][y+1], new_state[x][y]
    else:
        print("Movimento inválido!")
    return new_state

# ---------------- Interface ----------------

def main():
    start_state = [[2, 8, 3],
                   [1, 6, 4],
                   [7, 0, 5]]

    print("===== JOGO DOS OITO =====")
    print("Escolha o modo:")
    print("1 - Jogar manualmente")
    print("2 - Resolver automaticamente (A*)")

    choice = input("Digite sua escolha (1/2): ")

    # --- MODO 1: Jogador manual ---
    if choice == '1':
        current_state = copy.deepcopy(start_state)
        print("\nUse as teclas W (cima), S (baixo), A (esquerda), D (direita)")
        print("Objetivo:")
        print_board(goal_state)

        while True:
            print("Estado atual:")
            print_board(current_state)

            if current_state == goal_state:
                print("🎉 Parabéns! Você chegou ao objetivo!")
                break

            move_input = input("Seu movimento (W/A/S/D ou Q para sair): ").lower()
            if move_input == 'q':
                print("Jogo encerrado.")
                break

            current_state = move(current_state, move_input)

    # --- MODO 2: A* automático ---
    elif choice == '2':
        print("\nEscolha a heurística:")
        print("1 - Peças fora do lugar (h1)")
        print("2 - Distância de Manhattan (h2)")
        h_choice = input("Digite sua escolha (1/2): ")
        heuristic = 'misplaced' if h_choice == '1' else 'manhattan'

        print(f"\nResolvendo automaticamente com A* ({heuristic})...\n")
        start_time = time.time()
        path = a_star(start_state, heuristic=heuristic)
        end_time = time.time()

        if path:
            print(f"Solução encontrada em {len(path)-1} movimentos.")
            print(f"Tempo de execução: {end_time - start_time:.4f} segundos\n")
            for i, state in enumerate(path):
                print(f"Passo {i}:")
                print_board(state)
                time.sleep(0.5)
        else:
            print("Nenhuma solução encontrada.")

    else:
        print("Opção inválida!")

if __name__ == "__main__":
    main()
