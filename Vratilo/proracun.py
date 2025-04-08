#!/usr/bin/env python

import numpy as np

class Vratilo:
    def __init__(self, T_okretni, materijal, J_Z2, J_Z3,
                 l, G_Z2, G_Z3, b_Z2, b_Z3, r_Z2, r_Z3,
                 alpha_Z, beta_Z3) -> None:
        
        self.T_okretni = T_okretni # Moment vrtnje T [Nmm]
        self.materijal = materijal # Materijal vratila
        self.J_Z2 = J_Z2 # Moment tromosti Z2 - J2
        self.J_Z3 = J_Z3 # Moment tromosti Z3 - J3
        self.l = l # Razmak oslonaca A i B
        self.G_Z2 = G_Z2 # Tezina Z2
        self.G_Z3 = G_Z3  # Tezina Z3
        self.b_Z2 = b_Z2 # Sirina glavine Z2
        self.b_Z3 = b_Z3 # Sirina glavine Z3
        self.r_Z2 = r_Z2 # Diobeni polumjer Z2
        self.r_Z3 = r_Z3 # Diobeni polumjer Z3
        self.alpha_Z = alpha_Z # Standardni kut zahvatne linije
        self.beta_Z3 = beta_Z3 # Kut skosenja

        self.l3 = (self.l / 2) - (self.b_Z3) / 2 - 12
        self.l6 = (self.l / 2) + (self.b_Z2) / 2 + 12

        pass

    # Privatna metoda za zaokruzivanje vrijednosti u rjecniku na trazeni broj decimala
    def __roundDict(self, dict, decimals = 2) -> dict:
        return {key: np.round(value, decimals) for key, value in dict.items()}
    
    #Opterecenja na vratilu
    def opterecenjaNaV(self) -> dict:
        #Sile na Z2
        F_t2 = self.T_okretni / self.r_Z2 # Tangencijalna
        F_r2 = F_t2 * np.tan(self.alpha_Z) # Radijalna

        #Sile na Z3
        F_t3 = self.T_okretni / self.r_Z3
        F_r3 = F_t3 * (np.tan(self.alpha_Z)/np.cos(self.beta_Z3))
        F_a3 = F_t3 * np.tan(self.beta_Z3)

        M_a3 = F_a3 * self.r_Z3

        return self.__roundDict({"F_t2":F_t2, "F_r2":F_r2, "F_t3":F_t3, "F_r3":F_r3, "F_a3":F_a3, "M_a3":M_a3})
    
    #Reakcije u osloncima, postavljeno kao dva sustava lin. jdbi za horizontalnu(h) i vertikalnu (v) ravninu
    def reakcijeOslonci(self, opter : dict) -> dict:
        A_h = np.array([[1, 1], [0, self.l]])
        b_h = np.array([opter["F_r3"] - opter["F_r2"], opter["F_r3"] * self.l6 - opter["F_r2"] * self.l3-opter["M_a3"]])
        X_h = np.linalg.solve(A_h, b_h).round(2)

        A_v = np.array([[1, 1, 0], [0, self.l, 0], [0, 0, 1]])
        b_v = np.array([self.G_Z2 + opter["F_t2"] + self.G_Z3 + opter["F_t3"], self.l3 *(self.G_Z2 + opter["F_t2"]) + self.l6 * (self.G_Z3 + opter["F_t3"]), opter["F_a3"]])
        X_v = np.linalg.solve(A_v, b_v).round(2)
        
        return {"F_Ay":X_h[0], "F_By":X_h[1], "F_Az":X_v[0], "F_Bz":X_v[1], "F_Bx":X_v[2]}

    def Q_y_diagram(self, x, reak : dict, opter : dict):
        if not np.all((x >= 0) & (x <= self.l)):
            raise IndexError("Zadane tocke (x) nisu u intervalu [0, l]")
        y = np.zeros(len(x))
        y += reak["F_Ay"] * (x > 0) * (x <= self.l3)
        y += (reak["F_Ay"] + opter["F_r2"]) * (x > self.l3) * (x <= self.l6)
        y += -reak["F_By"] * (x > self.l6) * (x <= self.l)
        return y
    
def main():
    #Parametri
    T_okretni = 570_000 # Moment vrtnje T [Nmm]
    materijal = "0561" # Materijal vratila Č.0561
    J_Z2 = 0.0500 # Moment tromosti Z2 - J2
    J_Z3 = 0.875 # Moment tromosti Z3 - J3
    l = 370 # Razmak oslonaca A i B
    G_Z2 = 210 # Tezina Z2
    G_Z3 = 80  # Tezina Z3
    b_Z2 = 110 # Sirina glavine Z2
    b_Z3 = 110 # Sirina glavine Z3
    r_Z2 = 165 # Diobeni polumjer Z2
    r_Z3 = 57.8 # Diobeni polumjer Z3
    alpha_Z = np.deg2rad(20) # Standardni kut zahvatne linije
    beta_Z3 = np.deg2rad(18) # Kut skosenja

    vratilo = Vratilo(T_okretni, materijal, J_Z2, J_Z3, l, G_Z2, G_Z3, b_Z2, b_Z3, r_Z2, r_Z3, alpha_Z, beta_Z3)
    opterecenja = vratilo.opterecenjaNaV()
    print(opterecenja)
    print(vratilo.reakcijeOslonci(opterecenja))
    print(vratilo.Q_y_diagram(np.linspace(1, 370, 100), vratilo.reakcijeOslonci(opterecenja), opterecenja))

if __name__ == '__main__': 
    main()
        
        