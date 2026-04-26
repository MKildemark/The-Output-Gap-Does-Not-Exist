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
