# Models

## 1.1 Hasenzagel model

**Observation block**
$$
\underset{\mathrm{obs}}{
\begin{pmatrix}
y_t\\
e_t\\
u_t\\
oil_t\\
\pi_t\\
\pi_t^{c}\\
F_t^{uom}\pi_{t+4}\\
F_t^{spf}\pi_{t+4}
\end{pmatrix}}
=
\underset{\mathrm{obs.mat.\ }Z}{
\begin{pmatrix}
1 & 0 & 0\\
\delta_{e,1}+\delta_{e,2}L & 0 & 0\\
\delta_{u,1}+\delta_{u,2}L & 0 & 0\\
\delta_{oil,1}+\delta_{oil,2}L & 1 & 0\\
\delta_{\pi,1}+\delta_{\pi,2}L & \gamma_{\pi,1}+\gamma_{\pi,2}L & 1/\sigma_{obs}\\
\delta_{\pi^{c},1}+\delta_{\pi^{c},2}L & \gamma_{\pi^{c},1}+\gamma_{\pi^{c},2}L & 1/\sigma_{obs}\\
\delta_{uom,1}+\delta_{uom,2}L+\delta_{uom,3}L^{2} & \gamma_{uom,1}+\gamma_{uom,2}L & 1/\sigma_{obs}\\
\delta_{spf,1}+\delta_{spf,2}L+\delta_{spf,3}L^{2} & \gamma_{spf,1}+\gamma_{spf,2}L & 1/\sigma_{obs}
\end{pmatrix}}
\underset{\mathrm{states}}{
\begin{pmatrix}
\Psi_t\\
\Psi_t^{EP}\\
\mu_t^{\pi}
\end{pmatrix}}
+
\underset{\mathrm{idio.cycles}}{
\begin{pmatrix}
\Psi_t^{y}\\ \Psi_t^{e}\\ \Psi_t^{u}\\ \Psi_t^{oil}\\
\Psi_t^{\pi}\\ \Psi_t^{\pi^{c}}\\ \Psi_t^{uom}\\ \Psi_t^{spf}
\end{pmatrix}}
+
\underset{\mathrm{trends/noise}}{
\begin{pmatrix}
\mu_t^{y}\\ \mu_t^{e}\\ \mu_t^{u}\\ \mu_t^{oil}\\ 0\\ 0\\ \mu_t^{uom}\\ \mu_t^{spf}
\end{pmatrix}}
$$

Here, $\Psi$ are generalized ARMA(2,1) (Harvey) cycles; $\mu$ are random-walk trends (some with drift).

**Code formulation (expanded common state)**
$$
\underset{\mathrm{obs}}{
\begin{pmatrix}
y_t\\ e_t\\ u_t\\ oil_t\\ \pi_t\\ \pi_t^{c}\\ F_t^{uom}\pi_{t+4}\\ F_t^{spf}\pi_{t+4}
\end{pmatrix}}
=
\underset{\mathrm{obs.mat.\ }Z}{
\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0\\
\delta_{e,1} & \delta_{e,2} & 0 & 0 & 0 & 0\\
\delta_{u,1} & \delta_{u,2} & 0 & 0 & 0 & 0\\
\delta_{oil,1} & \delta_{oil,2} & 0 & 1 & 0 & 0\\
\delta_{\pi,1} & \delta_{\pi,2} & 0 & \gamma_{\pi,1} & \gamma_{\pi,2} & 1/\sigma_{obs}\\
\delta_{\pi^{c},1} & \delta_{\pi^{c},2} & 0 & \gamma_{\pi^{c},1} & \gamma_{\pi^{c},2} & 1/\sigma_{obs}\\
\delta_{uom,1} & \delta_{uom,2} & \delta_{uom,3} & \gamma_{uom,1} & \gamma_{uom,2} & 1/\sigma_{obs}\\
\delta_{spf,1} & \delta_{spf,2} & \delta_{spf,3} & \gamma_{spf,1} & \gamma_{spf,2} & 1/\sigma_{obs}
\end{pmatrix}}
\underset{\mathrm{common\ states}}{
\begin{pmatrix}
\Psi_t\\ \Psi_t^{*}\\ \Psi_{t-1}^{*}\\ \Psi_t^{EP}\\ \Psi_t^{EP*}\\ \mu_t^{\pi}
\end{pmatrix}}
+ \mathrm{idio.cycles} + \mathrm{idio.trends}.
$$

**State ordering in code**
$$
\big(\Psi_t,\Psi_t^{*},\Psi_{t-1}^{*},\Psi_t^{EP},\Psi_t^{EP*},\mu_t^{\pi},
\Psi_t^{y},\Psi_t^{y*},\mu_t^{y},\Psi_t^{e},\Psi_t^{e*},\mu_t^{e},
\Psi_t^{u},\Psi_t^{u*},\mu_t^{u},\Psi_t^{oil},\Psi_t^{oil*},\mu_t^{oil},
\Psi_t^{\pi},\Psi_t^{\pi*},\Psi_t^{\pi^{c}},\Psi_t^{\pi^{c}*},
\Psi_t^{uom},\Psi_t^{uom*},\mu_t^{uom},\Psi_t^{spf},\Psi_t^{spf*},\mu_t^{spf}\big).
$$

**Cycle dynamics (generic 2×2 rotation AR(1))**
$$
\begin{pmatrix} g_t\\ g_t^{*} \end{pmatrix}
=
\rho\!
\begin{pmatrix}
\cos\lambda & \sin\lambda\\
-\sin\lambda & \cos\lambda
\end{pmatrix}
\begin{pmatrix} g_{t-1}\\ g_{t-1}^{*} \end{pmatrix}
+
\begin{pmatrix} \varepsilon_t^{g}\\ \varepsilon_t^{g^{*}} \end{pmatrix}.
$$

---

## 1.2 Empirical Flex+Eff NK model with Hasenzagel expectations

### 1.2.1 Baseline Hasenzagel expectations with NKPC

$$
\underset{\mathrm{obs}}{
\begin{pmatrix}
y_t\\ e_t\\ u_t\\ \pi_t\\ F_t^{uom}\pi_{t+4}\\ F_t^{spf}\pi_{t+4}
\end{pmatrix}}
=
\mathrm{Diag}(1/\sigma)\,
\underset{\mathrm{obs.mat.\ }Z}{
\begin{pmatrix}
1 & 0 & 0 & 0 & 0\\
\delta_{e} & 0 & 0 & 0 & 0\\
\delta_{u} & 0 & 0 & 0 & 0\\
(\delta_E+\kappa) & 0 & (\gamma_E+1) & 0 & 1\\
\delta_E & 0 & \gamma_E & 0 & 1\\
\delta_E & 0 & \gamma_E & 0 & 1
\end{pmatrix}}
\begin{pmatrix}
g_t^{e}\\ g_t^{e*}\\ g_t^{\pi}\\ g_t^{\pi*}\\ \mu_t^{\pi}
\end{pmatrix}
+ \mathrm{idio.cycles} + \mathrm{idio.trends}.
$$

- $g$ and $\Psi$: Harvey cycles (ARMA(2,1)/trigonometric).  
- $\mu$: random walks; $\mu^{y},\mu^{e}$ include drift.  
- Innovations orthogonal **except** shocks to $(g^{e},g^{\pi})$ (allow correlation).  
- Data standardized; common states in original units; scales pinned by unity loadings and $\mathrm{Diag}(1/\sigma)$.  
- Expectations are reduced-form: $\delta_E,\gamma_E$ free; need not be rational.

---

### 1.2.2 Short-term expectations rational in NKPC

We impose that the **common component** shared by $\pi_t$ and short-term expectations ($h=4$) is **rational** with respect to $(g^{e},g^{\pi},\mu^{\pi})$. Each expectations series still has its own idiosyncratic cycle/trend.

**Definitions**
$$
\Phi_{e}=\rho_{e}
\begin{pmatrix}
\cos\lambda_{e} & \sin\lambda_{e}\\
-\sin\lambda_{e} & \cos\lambda_{e}
\end{pmatrix},\qquad
\Phi_{\pi}=\rho_{\pi}
\begin{pmatrix}
\cos\lambda_{\pi} & \sin\lambda_{\pi}\\
-\sin\lambda_{\pi} & \cos\lambda_{\pi}
\end{pmatrix},\qquad
s'=\begin{pmatrix}1 & 0\end{pmatrix}.
$$

**Derivation of current-inflation loadings $\alpha$.**  
Let $\tilde\pi_t=\pi_t-\mu_t^\pi$. The NKPC is
$$
\tilde\pi_t=\kappa\,x_t+\beta\,\mathbb{E}_t\tilde\pi_{t+1}+u_t,
\qquad x_t=s' g_t^{e},\quad u_t=s' g_t^{\pi}.
$$
Since $\mathbb{E}_t g^{\bullet}_{t+j}=\Phi_{\bullet}^{j}g_t^{\bullet}$,
$$
\tilde\pi_t
=\sum_{j=0}^{\infty}\beta^{j}\big(\kappa\,s'\Phi_e^{j}g_t^{e}+s'\Phi_\pi^{j}g_t^{\pi}\big)
\equiv \alpha_{e}'g_t^{e}+\alpha_{\pi}'g_t^{\pi},
$$
with
$$
\boxed{\;
\alpha_{e}'=\kappa\,s'(I_2-\beta\,\Phi_e)^{-1},\qquad
\alpha_{\pi}'=\,s'(I_2-\beta\,\Phi_\pi)^{-1}\;}
\quad (|\beta\rho_e|<1,\ |\beta\rho_\pi|<1).
$$

**Closed form (2×2 inverse).** For $a\in(-1,1)$,
$$
(I_2-a\,R(\lambda))^{-1}
=\frac{1}{1-2a\cos\lambda+a^2}
\begin{pmatrix}
1-a\cos\lambda & a\sin\lambda\\
-\,a\sin\lambda & 1-a\cos\lambda
\end{pmatrix}.
$$
Hence, writing $\alpha_{e}'=(\alpha_{e,1},\alpha_{e,2})$ and $\alpha_{\pi}'=(\alpha_{\pi,1},\alpha_{\pi,2})$,
$$
\alpha_{\pi,1}=\frac{1-\beta\rho_{\pi}\cos\lambda_{\pi}}{1-2\beta\rho_{\pi}\cos\lambda_{\pi}+(\beta\rho_{\pi})^{2}},
\qquad
\alpha_{\pi,2}=\frac{\beta\rho_{\pi}\sin\lambda_{\pi}}{1-2\beta\rho_{\pi}\cos\lambda_{\pi}+(\beta\rho_{\pi})^{2}},
$$
$$
\alpha_{e,1}=\kappa\,\frac{1-\beta\rho_{e}\cos\lambda_{e}}{1-2\beta\rho_{e}\cos\lambda_{e}+(\beta\rho_{e})^{2}},
\qquad
\alpha_{e,2}=\kappa\,\frac{\beta\rho_{e}\sin\lambda_{e}}{1-2\beta\rho_{e}\cos\lambda_{e}+(\beta\rho_{e})^{2}}.
$$

**$h$-ahead mapping.** Propagate states $h$ steps:
$$
\mathbb{E}_t\pi_{t+h}
=\mu_t^\pi+\alpha_{e}'\Phi_e^{h}g_t^{e}+\alpha_{\pi}'\Phi_\pi^{h}g_t^{\pi},
\qquad
\boxed{\;\alpha_{e}'(h)=\alpha_{e}'\Phi_e^{h},\quad \alpha_{\pi}'(h)=\alpha_{\pi}'\Phi_\pi^{h}\;}
$$
with $\Phi_{\bullet}^{h}=\rho_{\bullet}^{h}\begin{pmatrix}\cos(h\lambda_{\bullet})&\sin(h\lambda_{\bullet})\\-\sin(h\lambda_{\bullet})&\cos(h\lambda_{\bullet})\end{pmatrix}$.

**Measurement block (RE common part, $h=4$)**
$$
\underset{\mathrm{obs}}{
\begin{pmatrix}
y_t\\ e_t\\ u_t\\ \pi_t\\ F_t^{uom}\pi_{t+4}\\ F_t^{spf}\pi_{t+4}
\end{pmatrix}}
=
\mathrm{Diag}(1/\sigma)\,
\underset{\mathrm{obs.mat.\ }Z}{
\begin{pmatrix}
1 & 0 & 0 & 0 & 0\\
\delta_{e} & 0 & 0 & 0 & 0\\
\delta_{u} & 0 & 0 & 0 & 0\\
\alpha_{e,1} & \alpha_{e,2} & \alpha_{\pi,1} & \alpha_{\pi,2} & 1\\
\alpha_{e,1}(4) & \alpha_{e,2}(4) & \alpha_{\pi,1}(4) & \alpha_{\pi,2}(4) & 1\\
\alpha_{e,1}(4) & \alpha_{e,2}(4) & \alpha_{\pi,1}(4) & \alpha_{\pi,2}(4) & 1
\end{pmatrix}}
\begin{pmatrix}
g_t^{e}\\ g_t^{e*}\\ g_t^{\pi}\\ g_t^{\pi*}\\ \mu_t^{\pi}
\end{pmatrix}
+ \mathrm{idio.cycles} + \mathrm{idio.trends}.
$$

---

### 1.2.3 Long-term expectations rational in NKPC

Use a long-horizon expectation to anchor trend inflation while business-cycle NKPC dynamics are governed by the model. Impose RE on the **common** part of the long-term row(s) at horizon $H$ (e.g., $H=40$).

**RE loadings and horizon mapping**
$$
\alpha_{e}'=\kappa\,s'(I_2-\beta\Phi_e)^{-1},\qquad
\alpha_{\pi}'=s'(I_2-\beta\Phi_\pi)^{-1},\qquad
\alpha_{e}'(H)=\alpha_{e}'\,\Phi_e^{H},\quad
\alpha_{\pi}'(H)=\alpha_{\pi}'\,\Phi_\pi^{H}.
$$

**Specification (long-term only)**
$$
\underset{\mathrm{obs}}{
\begin{pmatrix}
y_t\\ e_t\\ u_t\\ \pi_t\\ F_t^{LT}\pi_{t+H}
\end{pmatrix}}
=
\mathrm{Diag}(1/\sigma)\,
\underset{\mathrm{obs.mat.\ }Z}{
\begin{pmatrix}
1 & 0 & 0 & 0 & 0\\
\delta_{e} & 0 & 0 & 0 & 0\\
\delta_{u} & 0 & 0 & 0 & 0\\
\alpha_{e,1} & \alpha_{e,2} & \alpha_{\pi,1} & \alpha_{\pi,2} & 1\\
\alpha_{e,1}(H) & \alpha_{e,2}(H) & \alpha_{\pi,1}(H) & \alpha_{\pi,2}(H) & 1
\end{pmatrix}}
\begin{pmatrix}
g_t^{e}\\ g_t^{e*}\\ g_t^{\pi}\\ g_t^{\pi*}\\ \mu_t^{\pi}
\end{pmatrix}
+ \mathrm{idio.cycles} + \mathrm{idio.trends}.
$$

**Specification B (long-term + reduced-form short-term)**  
Keep UoM/SPF rows as in §1.2.1 and impose RE **only** on the long-term row. This tightly anchors $\mu_t^\pi$ while leaving $\kappa$ primarily identified from $\pi_t$.

**Remark.** For large $H$, $\Phi^{H}$ damps cycles; the LT row mainly identifies $\mu_t^\pi$. If you want cycle information from LT expectations, use a moderate $H$ (e.g., $12$–$20$) or combine with §1.2.2.
