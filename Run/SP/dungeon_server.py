#!/usr/bin/python
import socket 
import select
import sys
import json

def message_data(sock, msg):
    for socket in CONNECTIONS:
        #notification in scoring
        if socket == sock and socket != server:
            try:
                socket.sendall(msg)
            except:
                socket.close()
                CONNECTIONS.remove(socket)

def shieldProcess(wizardMove, shieldNum):
    gamestate = 1
    if wizardMove == "1":
        message = "\nWizard planted a bomb."
    elif wizardMove == "2":
        message = "\nWizard used a bomb multiplier enchantment."
    if shieldNum > 0:
        message += " Pauper stepped on the trap but is protected by a shield! The game goes on!"
        data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
        #data = json.dumps(data)
        message_data(PLAYERS[0][0], data)
        message_data(PLAYERS[1][0], data)
        return True

    
    message += " Pauper tried using a shield but the item is broken! The trap activates!"
    if wizardMove == "1":
        gamestate = 2
        data1 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"]) + "\nGAME OVER\nYou Win!! :D"
        data2 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"]) + "\nGAME OVER for u Pauper :P"
        #data1 = json.dumps(data1)
        #data2 = json.dumps(data2)
        message_data(PLAYERS[0][0], data1)
        message_data(PLAYERS[1][0], data2)
    else:
        data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
        #data = json.dumps(data)
        message_data(PLAYERS[0][0], data)
        message_data(PLAYERS[1][0], data)
    return False

def swordProcess(wizardMove, swordNum):
    gamestate = 1
    if wizardMove == "1":
        message = "\nWizard planted a bomb."
    elif wizardMove == "2":
        message = "\nWizard used a bomb multiplier enchantment."
    if swordNum > 0:
        message += " Pauper stepped on the trap but attacks it with an enchanted sword! The number of bombs are halved!"
        data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
        #data = json.dumps(data)
        message_data(PLAYERS[0][0], data)
        message_data(PLAYERS[1][0], data)
        return True

    message += " Pauper tried to use the enchanted sword but failed! The trap activates!"
    if wizardMove == "1":
        gamestate = 2
        data1 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"]) + "\nGAME OVER\nYou Win!! :D"
        data2 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"]) + "\nGAME OVER for u Pauper :P"
        #data1 = json.dumps(data1)
        #data2 = json.dumps(data2)
        message_data(PLAYERS[0][0], data1)
        message_data(PLAYERS[1][0], data2)
    else:
        data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
        #data = json.dumps(data)
        message_data(PLAYERS[0][0], data)
        message_data(PLAYERS[1][0], data)
    return False

if __name__ == "__main__":
    # 0 - server, 1 - Player 1, 2 - Player 2
    CONNECTIONS = []
    RECV_BUFFER = 4096
    PORT = 5000
    HOST = socket.gethostname()
    PLAYERS = []
    WIZARD_MOVES = []
    PAUPER_MOVES = []
    LIMIT = 3
    WIZARD = {"bombs": 5, "multibomb": 2, "removeitem": 2}
    PAUPER = {"turns": 5, "sword": 2, "shield": 2}
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    #binding server to specified host and port
    server.bind((HOST, PORT))
    server.listen(2)

    #adds server to readable connections
    CONNECTIONS.append(server)

    # server will send signals to client using a dictionary
    # 0 is game is waiting for another player
    # 1 is game is starting/ongoing
    # 2 is game is finished with wizard winning
    # 3 is game is finished with pauper winning
    while True:
        print ("Waiting for connections...")
        conn, addr = server.accept()
        print ("Player (%s, %s) connected" % addr)
        CONNECTIONS.append(conn)
        PLAYERS.append([conn, addr, 0])
        if len(CONNECTIONS) == LIMIT:
            print ("The game is now starting\nWizard <"+str(PLAYERS[0][1])+">\nPauper <"+ str(PLAYERS[1][1])+">")
            gamestate = 1
            message = "The game is now starting\nWizard <You>\nPauper <"+str(PLAYERS[1][1])+">"
            #data = {"gameState" : gameState, "message": message}
            #data = json.dumps(data)
            data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
            message_data(PLAYERS[0][0], data)
            message = "The game is now starting\nWizard <"+str(PLAYERS[0][1])+">\nPauper <You>"
            message_data(PLAYERS[1][0], data)
            break
        else:
            gamestate = 0
            message = "Waiting for another player..."
            data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
            #data = json.dumps(data)
            message_data(conn, data)


    while WIZARD["bombs"] > 0:
        inputready, outputready, errorready = select.select(CONNECTIONS,[],[])

        #new client connection
        for sock in inputready:
            if sock == server:
                #handles new client
                c, addr = server.accept()
                CONNECTIONS.append(c)
                print ("Player <", addr, "> connected")
            else:
                try:
                    data = sock.recv(RECV_BUFFER)
                    data.split()
                    message = ""
                    if sock == PLAYERS[0][0]:
                        WIZARD_MOVES.append(data)
                    if sock == PLAYERS[1][0]:
                        PAUPER_MOVES.append(data)
                    if len(WIZARD_MOVES) >= 1 and len(PAUPER_MOVES) >= 1:
                        WIZARD["bombs"] = WIZARD["bombs"] - 1
                        wizard_move = WIZARD_MOVES.pop()
                        wizard_move = wizard_move.split( )
                        pauper_move = PAUPER_MOVES.pop()
                        pauper_move = pauper_move.split( )
                        gamestate = 1
                        message1 = "\nWizard's move:\n\tAction "+wizard_move[0]+"\n\tDoor "+wizard_move[1]
                        message2 = "\nPauper's move:\n\tAction "+pauper_move[0]+"\n\tDoor "+pauper_move[1]
                        data1 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                        data2 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                        message_data(PLAYERS[0][0], data)
                        #data1 = json.dumps(data)
                        #data2 = json.dumps(data)
                        message_data(PLAYERS[0][0], data1)
                        message_data(PLAYERS[0][0], data2)
                        message_data(PLAYERS[1][0], data1)
                        message_data(PLAYERS[1][0], data2)
                        #both players choose the same door
                        if wizard_move[1] == pauper_move[1]:
                            if wizard_move[0] == "1":
                                if pauper_move[0] == "1":
                                    gamestate = 2
                                    message = "\nGame over"
                                    data1 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                                    message_data(PLAYERS[0][0], data)
                                    data2 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                                    message_data(PLAYERS[0][0], data)
                                    #data1 = json.dumps(data)
                                   # data2 = json.dumps(data)
                                    message_data(PLAYERS[0][0], data1)
                                    message_data(PLAYERS[1][0], data2)
                                    server.close()
                                    sys.exit()
                                elif pauper_move[0] == "2":
                                    if(not shieldProcess("1", PAUPER["shield"])):
                                        server.close()
                                        sys.exit()
                                    PAUPER["shield"] = PAUPER["shield"] - 1
                                elif pauper_move[0] == "3":
                                    if (not swordProcess("1", PAUPER["sword"])):
                                        server.close()
                                        sys.exit()
                                    PAUPER["sword"] = PAUPER["sword"] - 1
                                    WIZARD["bombs"] = WIZARD["bombs"] / 2
                            elif wizard_move[0] == "2":
                                if WIZARD["multibomb"] > 0:
                                    if pauper_move[0] == "1":
                                        message = "\nPauper stepped on a bomb multiplier! The number of bombs has increased!"
                                        data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                                        message_data(PLAYERS[0][0], data)
                                        #data = json.dumps(data)
                                        message_data(PLAYERS[0][0], data)
                                        message_data(PLAYERS[1][0], data)
                                        WIZARD["multibomb"] = WIZARD["multibomb"] - 1
                                        WIZARD["bombs"] = WIZARD["bombs"] * 2
                                    elif pauper_move[0] == "2":
                                        shieldProcess("2", PAUPER["shield"])
                                        PAUPER["shield"] = PAUPER["shield"] - 1
                                    elif pauper_move[0] == "3":
                                        swordProcess("2", PAUPER["sword"])
                                        PAUPER["sword"] = PAUPER["sword"] - 1
                                    WIZARD["multibomb"] = WIZARD["multibomb"] - 1
                                else:
                                    message = "\nWizard tried using a bomb multiplier enchantment but failed! The game goes on!"
                                    data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                                    message_data(PLAYERS[0][0], data)
                                    #data = json.dumps(data)
                                    message_data(PLAYERS[0][0], data)
                                    message_data(PLAYERS[1][0], data)
                            elif wizard_move[0] == "3":
                                if WIZARD["removeitem"] != 0:
                                    message = "\nPauper stepped on a remove item enchantment! All items were removed!"
                                    data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                                    message_data(PLAYERS[0][0], data)
                                    #data = json.dumps(data)
                                    message_data(PLAYERS[0][0], data)
                                    message_data(PLAYERS[1][0], data)
                                    PAUPER["sword"] = 0
                                    PAUPER["shield"] = 0
                                    WIZARD["removeitem"] = WIZARD["removeitem"] - 1
                                else:
                                    message = "\nWizard tried using a remove item enchantment but failed! The game goes on!"
                                    data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"]) + "\nGAME OVER\nYou Win!! :D"
                                    #data = json.dumps(data)
                                    message_data(PLAYERS[0][0], data)
                                    message_data(PLAYERS[1][0], data)
                            # if shield == true: next level
                            # if sword == ture: numOfBombs/2
                            # if bombmultiplier == true: numOfBombs * 2
                            # if removeitem == true: pauperShield = 0, pauperSword = 0
                        elif WIZARD["bombs"] == 0:
                            gamestate = 3
                            message = "\nGame over"
                            data1 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                            message_data(PLAYERS[0][0], data)
                            data2 = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                            message_data(PLAYERS[0][0], data)
                            #data1 = json.dumps(data1)
                            #data2 = json.dumps(data2)
                            message_data(PLAYERS[0][0], data1)
                            message_data(PLAYERS[1][0], data2)
                        else:
                            message = "\nThe wizard and the pauper chose different doors. The game goes on!"
                            data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"]) + "\nGAME OVER\nYou Win!! :D"
                            data = json.dumps(data)
                            message_data(PLAYERS[0][0], data)
                            message_data(PLAYERS[1][0], data)
                        message = "\nBombs left: "+str(WIZARD["bombs"])
                        data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                        message_data(PLAYERS[0][0], data)
                        #data = json.dumps(data)
                        message_data(PLAYERS[1][0], data)
                        message_data(PLAYERS[0][0], data)
                        WIZARD_MOVES = []
                        PAUPER_MOVES = []
                    #=================================================

                except Exception as inst:
                    message = "Client (%s, %s) was disconnected... Closing the game..." % addr
                    gamestate = 4
                    data = str(gamestate)+"|"+message+"|"+str(WIZARD["bombs"])+"|"+str(WIZARD["multibomb"])+"|"+str(WIZARD["removeitem"])+"|"+str(PAUPER["sword"])+"|"+str(PAUPER["shield"])
                    message_data(PLAYERS[0][0], data)
                    #data = json.dumps(data)
                    message_data(sock, data)
                    print ("Client: ", addr, " is offline")
                    print (type(inst))
                    print (inst.args)
                    CONNECTIONS.rmove(sock)
                    break
    server.close()s