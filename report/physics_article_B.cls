\LoadClass[a4paper,11pt]{article}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5\baselineskip}%

\ProvidesClass{physics_article_B}[18/05/15 - Article class for reports written by Nottingham Physics students, without biblatex]


\RequirePackage{fancyhdr}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{textcomp} % Prevent error when loading gensymb
\usepackage{gensymb} % For degree symbol
\RequirePackage{commath}
\RequirePackage{bm}
\RequirePackage{graphicx}
\RequirePackage[hidelinks]{hyperref}

\RequirePackage{fontspec}
\setmainfont{Lora}[
  Path = Lora/,
  Extension = .ttf,
  UprightFont = *-Regular,
  BoldFont = *-Bold,
  ItalicFont = *-Italic,
  BoldItalicFont = *-BoldItalic]

\RequirePackage{cleveref}
\RequirePackage[backend=biber, style=nature]{biblatex}

\RequirePackage{geometry}
\geometry{
  a4paper,
  left=20mm,
  right=20mm,
  top=25mm,
  bottom=25mm
}
\RequirePackage{float}

\RequirePackage{wrapfig}
\RequirePackage[font=small,labelfont=bf]{caption}
\RequirePackage{subcaption}

\RequirePackage{color}
\RequirePackage{listings}
\RequirePackage{tabularx}
\RequirePackage{svg}
\RequirePackage{mhchem}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.16}

\definecolor{mygreen}{RGB}{28,172,0} % color values Red, Green, Blue
\definecolor{mylilas}{RGB}{170,55,241}

\lstset{language=Matlab,
    breaklines=true,
    morekeywords={matlab2tikz},
    keywordstyle=\color{blue},
    morekeywords=[2]{1},
    keywordstyle=[2]{\color{black}},
    identifierstyle=\color{black},
    stringstyle=\color{mylilas},
    commentstyle=\color{mygreen},
    showstringspaces=false,
    numbers=left,
    numberstyle={\tiny \color{black}},
    numbersep=9pt,
    emph=[1]{for,end,break},
    emphstyle=[1]\color{red},
}

% \renewcommand{\familydefault}{cmss}

\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\pagestyle{fancy}

\def\studentid#1{\gdef\@studentid{#1}}
%\def\myname#1{\gdef\@myname{#1}}
\def\mytitle#1{\gdef\@mytitle{#1}}

\lhead{ID: \@studentid}
\chead{}
\rhead{\leftmark}
\lfoot{}
\cfoot{\thepage}
\rfoot{}

\title{\@mytitle\vspace{-1.5cm}}

