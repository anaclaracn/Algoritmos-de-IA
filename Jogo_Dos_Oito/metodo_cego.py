from collections import deque
import copy
import time

# Estado objetivo
goal_state = [[1, 2, 3],
              [8, 0, 4],
              [7, 6, 5]]

# Função para encontrar a posição do zero (vazio)
def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

# Gera os estados vizinhos possíveis (movendo o zero)
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

# Converte o estado em tupla para usar em conjuntos
def state_to_tuple(state):
    return tuple(item for row in state for item in row)

# Verifica se o estado é o objetivo
def is_goal(state):
    return state == goal_state

# Busca em largura (BFS)
def bfs(start_state):
    queue = deque([(start_state, [])])  # (estado, caminho)
    visited = set()
    visited.add(state_to_tuple(start_state))
    expanded_nodes = 0  # contador de nós expandidos

    while queue:
        current_state, path = queue.popleft()
        expanded_nodes += 1  # incrementa a cada nó retirado da fila

        if is_goal(current_state):
            return path + [current_state], expanded_nodes

        for neighbor in get_neighbors(current_state):
            t = state_to_tuple(neighbor)
            if t not in visited:
                visited.add(t)
                queue.append((neighbor, path + [current_state]))
    return None, expanded_nodes

# Função para exibir o tabuleiro
def print_board(state):
    for row in state:
        print(" ".join(str(x) if x != 0 else "_" for x in row))
    print()

# Movimenta o espaço vazio conforme entrada do usuário
def move(state, direction):
    x, y = find_zero(state)
    new_state = copy.deepcopy(state)

    if direction == 'w' and x > 0:  # cima
        new_state[x][y], new_state[x-1][y] = new_state[x-1][y], new_state[x][y]
    elif direction == 's' and x < 2:  # baixo
        new_state[x][y], new_state[x+1][y] = new_state[x+1][y], new_state[x][y]
    elif direction == 'a' and y > 0:  # esquerda
        new_state[x][y], new_state[x][y-1] = new_state[x][y-1], new_state[x][y]
    elif direction == 'd' and y < 2:  # direita
        new_state[x][y], new_state[x][y+1] = new_state[x][y+1], new_state[x][y]
    else:
        print("Movimento inválido!")
    return new_state

# ---------------- INTERFACE ----------------
def main():
    start_state = [[2, 8, 1],
                  [0, 4, 3],
                  [7, 6, 5]]

    print("===== JOGO DOS OITO =====")
    print("Escolha o modo:")
    print("1 - Jogar manualmente")
    print("2 - Resolver automaticamente (Busca em Largura - BFS)")

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

            if is_goal(current_state):
                print("🎉 Parabéns! Você chegou ao objetivo!")
                break

            move_input = input("Seu movimento (W/A/S/D ou Q para sair): ").lower()
            if move_input == 'q':
                print("Jogo encerrado.")
                break

            current_state = move(current_state, move_input)

    # --- MODO 2: Resolução automática (BFS) ---
    elif choice == '2':
        print("\nResolvendo automaticamente com Busca em Largura (BFS)...")
        start_time = time.time()
        path, expanded_nodes = bfs(start_state)
        end_time = time.time()

        if path:
            print(f"\nSolução encontrada em {len(path)-1} movimentos.")
            print(f"Tempo de execução: {end_time - start_time:.4f} segundos")
            print(f"Nós expandidos: {expanded_nodes}\n")
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
