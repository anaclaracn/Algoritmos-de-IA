import heapq
import copy
import time
from itertools import count

# Estado objetivo
estado_objetivo = [[1, 2, 3],
                   [8, 0, 4],
                   [7, 6, 5]]
               

# ---------------- Funções de apoio ----------------

# Encontrar a posição do zero (vazio)
def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

# Converte o estado do tabuleiro (lista de listas mutável) em uma tupla (estrutura imutável)
# necessário para que os estados possam ser armazenados no conjunto (set) visited
def state_to_tuple(state):
    return tuple(item for row in state for item in row)

# Exibir o tabuleiro
def print_board(state):
    for row in state:
        print(" ".join(str(x) if x != 0 else "_" for x in row))
    print()

# Gera os estados vizinhos possíveis (movendo o zero) a partir do estado atual
def get_neighbors(state):
    neighbors = []
    x, y = find_zero(state)  # nova posição do zero
    moves = [(-1,0), (1,0), (0,-1), (0,1)]  # cima, baixo, esquerda, direita

    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = copy.deepcopy(state)
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(new_state)
    return neighbors

# ---------------- Heurística - Misplaced -> heurística das peças fora do lugar ----------------

def h_misplaced(state):
    count = 0
    # contabiliza quantas peças do tabuleiro não estão em sua posição correta em relação ao estado objetivo
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != estado_objetivo[i][j]:
                count += 1
    return count

# ---------------- Algoritmo A* ----------------

def a_star(estado_inicial):
    open_list = [] # Fila de Prioridade
    counter = count()  # contador incremental para desempate
    # Adiciona o estado inicial à fila
    heapq.heappush(open_list, (0, next(counter), estado_inicial, [])) # (custo total (f), contador, estado, caminho)
    visited = set() # conjunto de estados visitados
    visited.add(state_to_tuple(estado_inicial))
    nodes_expanded = 0

    while open_list:
        cost, _, current_state, path = heapq.heappop(open_list)
        nodes_expanded += 1

        if current_state == estado_objetivo:
            return path + [current_state], nodes_expanded

        g = len(path) # custo do caminho até o estado atual (custo real)

        for neighbor in get_neighbors(current_state):
            t = state_to_tuple(neighbor)
            if t not in visited:
                visited.add(t)

                h = h_misplaced(neighbor) # número de movimentos do neighbor até o estado_objetivo

                # +1: custo de um único movimento que leva do current_state para o neighbor
                f = g + 1 + h
                heapq.heappush(open_list, (f, next(counter), neighbor, path + [current_state]))

    return None, nodes_expanded

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
    estado_inicial = [[2, 8, 1],
                      [0, 4, 3],
                      [7, 6, 5]]

    print("===== JOGO DOS OITO =====")
    print("Escolha o modo:")
    print("1 - Jogar manualmente")
    print("2 - Resolver automaticamente (A*)")

    choice = input("Digite sua escolha (1/2): ")

    # --- MODO 1: Jogador manual ---
    if choice == '1':
        current_state = copy.deepcopy(estado_inicial)
        print("\nUse as teclas W (cima), S (baixo), A (esquerda), D (direita)")
        print("Objetivo:")
        print_board(estado_objetivo)

        while True:
            print("Estado atual:")
            print_board(current_state)

            if current_state == estado_objetivo:
                print("Parabéns! Você chegou ao objetivo!")
                break

            move_input = input("Seu movimento (W/A/S/D ou Q para sair): ").lower()
            if move_input == 'q':
                print("Jogo encerrado.")
                break

            current_state = move(current_state, move_input)

    # --- MODO 2: A*  ---
    elif choice == '2':
        start_time = time.time()
        path, nodes_expanded = a_star(estado_inicial)
        end_time = time.time()

        if path:
            print(f"Solução encontrada em {len(path)-1} movimentos.")
            print(f"Tempo de execução: {end_time - start_time:.4f} segundos")
            print(f"Número de estados visitados: {nodes_expanded}\n")
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
