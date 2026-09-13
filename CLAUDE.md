# CLAUDE.md — arbeidsnotater

Connectome-drevet C. elegans-simulering med pygame-grafikk. `README.md` forklarer
prosjektet; denne fila inneholder det som *ikke* kan leses ut av koden — hvorfor
konstantene er som de er, hva som er målt, og hva som allerede er prøvd og
forkastet.

## Kjøre

```bash
.venv/bin/python main.py              # vindu
.venv/bin/python main.py --headless 300
.venv/bin/python validate.py          # 8 sjekker, ~4 min
```

Venv finnes allerede (`numpy`, `pygame-ce`, Python 3.14). `validate.py` er
sannheten — den reproduserer hver måling som påstås i README. Kjør den etter
enhver endring i `neural.py`, `sensing.py` eller `body.py`.

**Kjøretid:** en simulert time tar ~2 min. Sveip over parametre tar fort >120 s
og bør kjøres i bakgrunnen.

## Datakilden

`3BIM20162017/CElegansTP` er **ren data, ingen kode** — ikke let etter en
simulering å porte, den finnes ikke. Fire CSV-er i `data/`: 299 nevroner / 2279
synapser, 118 motornevroner → 94 muskler, funksjonsmerking for 86 sensoriske
nevroner, 3D-posisjoner.

Fallgruver i dataene:
- AP-aksen (kolonne `1`) er **mest negativ i hodet**. Jeg antok motsatt først;
  `body_fraction()` har det riktig nå, og `validate.py` sjekker IL1L < AVM < PLML.
- 10 noder som starter med `M` (M1, M2L, MCL, MI…) er faryngeale *nevroner*, ikke
  muskler. Kroppsveggmusklene finnes bare i `Neurons_to_Muscles.csv`.
- Bare 9 % av synapsene er inhibitoriske, og alt annet er merket `exc` — inkludert
  de glutamat-styrte kloridsynapsene som egentlig er hemmende.

## Arkitektur

```
verden ─▶ sensing.py ─▶ neural.py ─▶ body.py ─▶ verden
        (sensorstrøm)  (299 LIF)   (kommando-  (bevegelse,
                                    avlesning)  kollisjon, mat)
```

`simulation.py` binder sammen. Nevralt steg er 1 ms; atferdssløyfa går på
framerate og tar så mange nevrale steg som får plass (rest bæres over i
`_neural_debt`).

## Bærende konstanter — ikke rør uten å måle

**`neural.py`.** `g_syn=150, i_tonic=28, noise=18` gir 10.9 ± 1.9 Hz over seeds.
Dette er et smalt vindu. Nettverket er **bistabilt** uten to ting:
- `normalize_weights=True` deler W på spektralradius (26.02), så `g_syn` blir
  løkkeforsterkningen i stedet for å avhenge av rå synapse-tall.
- `adapt_step=8, tau_adapt=120` — spike-frekvens-adaptasjon. Uten den løper
  det 91 % eksitatoriske nettet løpsk til 100–300 Hz, og all selektivitet dør.

`g_global` (global inhibisjon) finnes som parameter men står på 0 — normalisering
+ adaptasjon holder alene. Skala-intuisjon: rå `g_syn` måtte vært ~15 *før*
normalisering; mine første forsøk med 0.6–3.5 ga et helt stille nettverk.

**`body.py`.** `reversal_threshold=0.085` ligger mellom spontan maks (0.068) og
nese-berøring (~0.12). `tau_balance=0.45` er nødvendig — den øyeblikkelige
balansen har std ~0.065 rundt null, så uglattet blir membranstøy til konstante
reverseringer.

**`sensing.py`.** `pirouette_gain=350` gir 92 % over terskel ved fallende
gradient, 0 % ved stigende. `nose_gain=200` (var 90, ga bare 1 reversering på 347
kontakt-frames — for tynn margin).

## Hva som følger av connectomet, og hva som ikke gjør det

Dette er prosjektets faktiske funn og bør ikke utvannes.

**Emergent:** nese-berøring → revers (`gpg-nose` → AVA/AVD/AVE vekt 104 mot 26 på
AVB/PVC; målt balanse 0.119 mot 0.009 i hvile). Hele muskelutgangen. Hodemotorikkens
D/V-skille (SMDD 15:3 dorsalt, SMDV 0:10 ventralt, alle i segment 1–8).

**Ikke emergent — kjemotaksi.** Målt, ikke antatt:
- Mat-/luktnevroner har **0.000** direkte normalisert innflytelse på
  kommando-interneuronene.
- ASEL og ASER har nesten identiske målprofiler (begge → AIY, AIB), så ON/OFF-
  opponens finnes ikke i fila.
- Drev jeg ASE/AWC direkte, flyttet hodemuskel-ubalansen seg 0.006 mot et
  støynivå på 0.008.

Derfor er to regler skrevet ut eksplisitt i `sensing.py`, hver dokumentert der den
brukes, og hver leverer til connectomet for siste hopp:
- **Piruett** → AIZ + RIB (eneste interneuroner med entydig bakover-bias:
  AIZ→AVA 10 mot AVB 0; RIB 14 mot 1). **AIB er lærebok-ruta men er
  forover-biased her (4 mot 8)** — ikke bytt til den uten å måle.
- **Klinotaksi** → RIAs produktregel (sensorisk signal × hodeutslag) injisert i
  SMD/RMD dorsalt eller ventralt. Korrelasjon drive↔styringsavlesning: +0.78.

**Kalibreringen som må stå:** `Locomotion.calibrate()` måler hvilende D/V-ubalanse
(~0.14) på et ustimulert nettverk. Fjernes den, krøller sporet seg til en sirkel
på ~én kroppslengde og kjemotaksien forsvinner helt.

## Nevronkart, ablasjon og berikelse

`inspector.py` er et fullskjerms-overlegg (`tab`): kart over alle 299 øverst,
detaljpanel i tre kolonner nederst.

**Det viktigste designprinsippet her: de tre kolonnene holdes atskilt.**

| Kolonne | Kilde | Modul |
|---|---|---|
| LITERATURE | ekstern C. elegans-kunnskap, *ikke* fra datasettet | `annotations.py` |
| MEASURED | regnet ut fra CSV-ene | `analysis.py` |
| WHY ACTIVE NOW | live tilstand fra nettverket | `neural.input_breakdown()` |

Bland dem aldri i én visning. Resten av prosjektet er målt og etterprøvbart via
`validate.py`; annotasjonene er påstander. Skillet er det som gjør at man fortsatt
kan se hva connectomet faktisk gir.

`annotations.py` dekker 299/299 nevroner via 117 klasser. Klasseoppslaget er
`class_of()`, som prøver hele navnet *først* og så stripper gradvis — blind
stripping ødelegger AVL→AV, AQR→AQ, PQR→PQ, PVR→PV, PVD→PV, AFD→AF, RID/RIR/RIV→RI
og RMDL→RM. Legger du til klasser, kjør dekningssjekken på nytt.

Ablasjon lever i `NeuralNetwork.ablated`: nevronet fyrer ikke og `syn` nulles, så
det slutter å drive nedstrøms, men W står urørt. `stimulated` er motsatsen og
injiserer `stim_current`. Se `set_ablated` / `set_stimulated` / `restore_all`.

**Ikke gjør ablasjon til vektmaskering** (å nulle rader/kolonner i W) — da mister
du at nevronet fortsatt *mottar* input, og spektralradius-normaliseringen er
regnet ut fra hele W én gang i `__init__`.

`_rects` i inspektøren fylles først under `draw()`. Klikk før første tegning gir
`None`, som håndteres — men husk det hvis du skriver tester.

**Fargeskalaen i kartet må stå fast (`SCALE_HZ = 40`).** Den var normalisert til
97-persentilen, og det ga en direkte feiltolkning: ablerer du AVA (rangert #1–2)
faller persentilen 29.2 → 18.8 Hz, og alt annet tegnes ~35 % varmere uten å ha
endret seg. Det ser ut som kompensering. Ikke gjeninnfør adaptiv skalering.
`e` kjører et parvis eksperiment i stedet (`experiment.py`).

**Baseline alene er ikke nok, og `net.rate` duger ikke som kilde.** `tau_rate` er
60 ms, så ett aksjonspotensial flytter `rate` 16.7 Hz og 93 % av cellene avviker
fra et øyeblikksbilde uten at noe er endret. Inspektøren holder derfor sin egen
`smoothed` (tau 3 s) — ikke senk `tau_rate` i modellen, den er rask med vilje
fordi `Locomotion` leser den.

Selv glattet flagges 114/299 celler i en fritt bevegende orm uten manipulasjon:
ormens egen atferd varierer like mye som lesjonen. Lengre snitt hjelper ikke nok
(tau 20 s / 60 s vent: fortsatt 19 % null mot 37 % lesjon).

To løsninger, begge nødvendige:

1. **`Simulation.probe_step()`** — fryser kropp og sensorisk input, kun nettverket
   går. Null faller fra 114/299 til 1/299. `p` i UI, `release()` for å slippe.
2. **`experiment.measure()`** — det som faktisk brukes. Kjører to armer fra samme
   deepcopy-tilstand, samme frosne input og **samme RNG-seed**, som skiller seg
   bare i ablasjonen. Differansen er da ren årsak.

**Fallgruve jeg gikk i to ganger:** ikke les `net.rate` på slutten av en arm for å
sammenligne — tidskonstanten er 60 ms, så ett sample bærer ±8 Hz støy og drukner
effekten. `_run()` teller aksjonspotensialer over hele vinduet i stedet. Første
forsøk uten dette ga «nese-ablasjon hever AVAR +39.5 Hz», som er tull.

`validate.py` sjekker at ablasjon faktisk biter: nese-kretsen fjernet tar
berøringsbalansen fra 0.119 til −0.009.

## Ablasjon av AVA gir ingen hyperaktivering (målt)

Spørsmål som kommer igjen: ablerer man AVAL/AVAR/AIBL/AIBR ser AVEL, RIB m.fl. ut
til å bli hyperaktive. Målt i probe-modus: **92 celler ned, 0 opp.** AVEL går selv
*ned*. De store fallene er A-type bakover-motornevroner (DA6 −35, VA8 −33,
PVCL −26), som forventet når bakover-kommandoen fjernes.

De fire cellene har dessuten **null inhibitorisk utgang** (alle 370 i utgående
vekt er eksitatorisk), så de kan ikke disinhibere noe direkte. RIB stiger svakt i
lukket sløyfe, og da via to hopp: RIS (RIBs sterkeste hemmer, −3/−5) faller, som
slipper RIB opp.

Illusjonen kom fra adaptiv fargeskala + atferdsdrift. Se avsnittet over.

## Hvorfor noen nevroner fyrer mye (målt)

Fyringsrate korrelerer **+0.86** med signert netto inn-vekt, +0.84 med
eksitatorisk inn-vekt, +0.69 med inn-grad. Grunnen til at det er så rent: alle
nevroner har identiske indre egenskaper i modellen, så all variasjon *er*
koblingene. AVAR topper med 225 i eksitatorisk inn-vekt over 49 partnere.

Bunnen er sensoriske nevroner med 0–2 innkommende synapser (PLML har null) — de
er nettverkets inngang og skal drives av verden. 14 nevroner har ingen
innkommende synapse i det hele tatt.

Dette er delvis et artefakt: ekte nevroner har svært ulik eksitabilitet og er
stort sett graderte. Det er argumentet for gradert modell (se Neste steg).

**De 20 faryngeale nevronene er fullstendig frakoblet** — 0 synapser krysser
grensen i noen retning. I ekte ormer finnes broen RIP→I1; den mangler i fila. De
fyrer i kartet, men kan ikke påvirke atferd.

## Tidssteg — invariant som må stå

`Simulation.step()` deler tiden i skiver på maks `MAX_SUBSTEP = 0.02 s` og kaller
`_step_once()` per skive. **Ikke fjern det.** Atferdssløyfa er full av
forward-Euler-filtre med korteste tidskonstant 0.25 s (`tau_steer`). Uten
skiving ga `time_scale = 8` pluss én treg frame dt = 0.8 s, og `_steer`
eksploderte til −6e+19. Filtrene har i tillegg `min(1.0, dt/tau)` som
andrelinjeforsvar, siden `validate.py` kaller `Locomotion.update` direkte.

Nevrale steg er upåvirket — de kjører alltid på fast `neural_dt = 1 ms` via
`_neural_debt`.

## Visning: aldri autoskaler

Tre steder normaliserte til egen maksverdi og løy dermed om absoluttnivå. Alle er
nå faste: `SCALE_HZ = 40` (nevronkart), `ODOUR_REFERENCE = 1.0` (luktfelt),
`MUSCLE_REFERENCE = 3.2` (muskelgraf, målt p95). Tallene står i cellene uansett,
så ingenting går tapt. Ikke gjeninnfør adaptiv skalering noe sted.

## Blindveier — allerede testet, ikke gjenta

| Forsøk | Resultat |
|---|---|
| Tregt trendfilter (tau 2.5/9.0) til piruett-beslutningen | reversering 30 %, men matsøk 6/6 → 1/4. Parametrene står igjen i `SensoryParams` og vises i HUD, men brukes bevisst ikke. |
| Kortere reversering (1.0–1.4 s) | matsøk 0/4. Raten *stiger* (opptil 19/min) fordi ormen gjenopptar samme kurs og trigger på nytt. |
| Lengre refraktærperiode (2.5–8 s) | reversering ned til 8 %, matsøk 0/4. |
| Kraftigere omega-sving + kort reversering | best 2/4. |
| Høyere `steering_gain` (0.2–1.0) | ormen spiralerer på stedet, nærmeste avstand til mat 330+ av 350. |
| `dc_scale` opp til 0.045–0.07 | realistisk reverseringsrate, men spiser 0.00. |

Konklusjon: den lange reverseringen (2.2 s) er prisen for at kjemotaksien virker.
41 % tid i revers er valgt bevisst framfor en orm som ikke finner mat.

## Kjente svakheter

- **41 % tid i revers** mot 5–10 % hos ekte orm (raten på ~11/min er derimot
  innenfor lokalsøk). Se tabellen over.
- **Ingen area-restricted search** — reverseringsraten er lik på og utenfor mat
  (målt 37 % mot 40 %). Ekte ormer bytter atferdstilstand ved matfunn.
- Undulasjonen er en CPG i `body.py`, ikke emergent. LIF-nettet har verken gap
  junctions eller proprioseptiv tilbakekobling og *kan* ikke oscillere selv.

## Neste steg

Det klart mest verdifulle: **bytt datasett** til ett med gap junctions og riktige
fortegn (Cook 2019, WormAtlas, OpenWorm). Da kan piruett- og klinotaksi-reglene
fjernes og testes mot om atferden faller ut av koblingene alene. Bare
`sensing.py` må endres — `connectome.py` leser allerede alt inn generisk.

Mindre ting: metthet/sult som modulerer atferd (gir area-restricted search),
termotaksi og oksygen-taksi (`Sensory.csv` har cellene: AFD, URX/AQR/PQR),
proprioseptiv kobling i `body.py` slik at undulasjonen kan bli emergent.
