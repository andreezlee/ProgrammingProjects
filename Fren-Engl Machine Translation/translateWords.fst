#preprocess sentence for usage
define toLowercase A -> a , B -> b , C -> c , D -> d , E -> e , F -> f , G -> g , H -> h , I -> i , J -> j , K -> k , L -> l , M -> m , N ->  n , O -> o , P -> p , Q -> q , R -> r , S -> s , T -> t , U -> u , V -> v , W -> w , X -> x , Y -> y , Z -> z;
define removeContractions [du|des] -> [de " " le] , [au|aux] -> [à " " le] || [" " | .#. ] _ [" " | .#.];
define digits ["0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"];
define numbers [digits+ ["." digits+]^<2 ];
define punctuation2 "'" -> [e " "] .o. "." -> " ." , "?" -> " ?" || _ .#.;
define punctuation1 "," -> "." , "." -> "," || digits+ _ digits+;
define punctuation3 "," -> " ," || _ " " .o. ["-t-" | "-"] -> " " .o. "/" -> " / " || ?* _ ?*;


#TRANSLATIONS

#expression translations
define Expressions1 [plus " " de] -> [more " " than] , [alors " " que] -> [while] , [selon] -> [according " " to] , [face " " à] -> [opposite] , [plus " " tard] -> [later] , [à " " la " " fois] -> [at "-" the "-" same "-" time] , [est " " ce " " qui] -> 0 , [vis " " à " " vis] -> ["regarding/opposite"] || [.#. | " "] _ [.#. | " "] .o. [se " " | ne " " ] -> 0 || [.#. | " "] _;
define Expressions2 [le] -> 0 || [que] " " _ " " [on] " " ;
define Expressions3 [ce " " [qui | que]] -> [what] || [.#. | " "] _ [.#. | " "];
define Expressions4 [de " " [le|la]] -> "some" || [par | de | dans | jusque | à | pour | en | contre | avec | vers | sans | sur | entre | depuis | que | qui | lorsque | pourquoi | où | comment] " " _ [.#. | " "];
define Expressions Expressions1 .o. Expressions2 .o. Expressions3 .o. Expressions4;

#individual word translations - verbs
define justVerbs [ [rendre] | [combattre] | [être] | [concurrencer] | [passer] | [suivre] | [rester] | [parvenir] | [faire] | [trouver] | [a] | [ont] | [interpellent] | [confirmons] | [emparent] | [consacre] | [incarne] | [peut] | [peuvent] | [va] | [est] | [doit] | [affecte] | [gère] | [avait] | [avaient] | [stabilisait] | [faisait] | [profitaient] | [pourrait] | [chercherait] | [aura] ];

define objectPronouns [le | la] -> [it], [les | leur] -> [them] || [.#. | " "] _ [.#. | " "] justVerbs [.#. | " "] .o. [[eux | elles] -> them , cela -> that];

define Infinitives 
[rendre] -> "to render" , [combattre] -> "to fight" , [être] -> "to be" , [concurrencer] -> "to compete with" , [passer] -> "to pass/happen" , [suivre] -> "to pursue" , [rester] -> "to rest" , [faire] -> "to do/make" , [trouver] -> "to find" ,  [parvenir] -> "to reach" || [.#. | " "] _ [.#. | " "];

define ParticipleEndings [0 | e | s | "es"];
define Participles [réitéré [ParticipleEndings] ] -> [reiterated] , [enterré [ParticipleEndings] ] -> [buried] , [été [ParticipleEndings] ] -> [been] , [retrouvé [ParticipleEndings] ] -> [recovered] , [assassiné [ParticipleEndings] ] -> [assassinated] , [pris [ParticipleEndings] ] -> [taken] , [détourné [ParticipleEndings] ] -> [detained] , [annoncé [ParticipleEndings] ] -> [announced] , [remis [ParticipleEndings] ] -> [returned] , [indiqué [ParticipleEndings] ] -> [indicated] , [trouvé [ParticipleEndings] ] -> [found] , [envolé  [ParticipleEndings] ] -> [flown] , [opéré [ParticipleEndings] ] -> [operated "-" on] , [compliqué [ParticipleEndings] ] -> [complicated] , [redevenu [ParticipleEndings] " " normal] -> ["back-to-normal"] , [expliqué [ParticipleEndings] ] -> [explained] , [fermé [ParticipleEndings] ] -> [closed] , [épris [ParticipleEndings] ] -> ["in-love with"] , [enragé [ParticipleEndings] ] -> [enraged] , [fait " " pencher] -> [tipped] , [fait [ParticipleEndings] ] -> ["done/made"] , [interdisant [ParticipleEndings] ] -> [forbidding] || [.#. | " "] _ [.#. | " "];

define ConjugatedVerbs 
[a] -> "has" , [ont] -> "have" , [interpellent] -> "address" , [confirmons] -> "confirm" , [emparent] -> "take-possession" , [consacre] -> "blesses" , [incarne] -> "embodies" , [peut | peuvent] -> "can" , [va] -> "is going to" , [est] -> "is" , [doit] -> "must" , [affecte] -> "affect" , [gère] -> "deal-with" , [avait | avaient] -> "had" , [stabilisait] -> "stabilized" , [faisait] -> "did/made" , [profitaient] -> "profited" , [pourrait] -> "could" , [chercherait] -> "will search for" , [aura] -> "will have" , [était] -> [was] || [.#. | " "] _ [.#. | " "];

define Verbs objectPronouns .o. Infinitives .o. Participles .o. ConjugatedVerbs;



#individual word translations - nouns
define subjectPronouns [elle] -> ["she/her/it"] , [il] -> ["he/it"] , [nous] -> ["we/us"] , [on] -> ["one/you"] , [ils] -> [they] || [.#. | " "] _ [" " | .#.];

define properNouns 
[onu] -> ["United-Nations"] , [la " " france] -> [France] , [youssouf " " bakayoko] -> ["Youssouf-Bakayoko"] ,
 [avril] -> [April] , [ciudad " " juarez] -> ["Ciudad-Juarez"] , [aly " " zoulfecar] ->  ["Aly-Zoulfecar"] , [dar " " es " " salam] -> ["Dar-Es-Salam"] , [union " " européenne] -> ["European-Union"] , [tanzanie] -> [Tanzania] , [atalante] -> ["Operation-Atalanta"] , [sagrada " " familia] -> ["Sagrada-Familia"] , [liu " " xiaobo] -> ["Liu-Xiaobo"] , [chine] -> [China] , [bogdan " " klich] -> ["Bogdan-Klich"] , [varsovie] -> [Warsaw] , [new " " york] -> ["New-York"] , [novembre] -> [November] , [cristina " " fernandez " " de " " kirchner] -> ["Cristina-Fernandez-de-Kirchner"] , [m " . " deighton] -> "Mr.-Deighton" || [.#. | " "] _ [.#. | " "] .o. [mercredi] -> [Wednesday] , [samedi] -> [Saturday] , [vendredi] -> [Friday] , [jeudi] -> [Thursday] , [washington] -> [Washington] , [oslo] -> [Oslo] , [f " " 16] -> ["F-16"] , [hercules] -> [Hercules] , [marcelo] -> [Marcelo] , [abdallah] -> [Abdullah] , ["m."] -> ["Mr."] , [deighton] -> [Deighton] , [wikileaks] -> [WikiLeaks] , [cia] -> [CIA] || [.#. | " "] _;

define normalNouns 
[candidat] -> [candidate] , [voix] -> [voice] , [président [0 | e]] -> [president] , [soirée] -> [evening] , [résultat] -> [result] , 
[corps] -> [bodies] , [hommes] -> [men] , [homme] -> [man] , [femmes] -> [women/wives] , [femme] -> ["woman/wife"] , [victime] -> [victim] , 
[engagement] -> [commitment] , [terrorisme] -> [terrorism] , [dirigeant [0 | e]] -> [leader] , [pays] -> ["nation(s)"] , 
[monnaies] -> [currencies] , [monnaie] -> [currency] , [yen [0 | s]] -> [yen] , [soir] -> [night] , [mort] -> [death] , 
[citoyen] -> [citizen] , [année] -> [year] , [rivalité] -> [rivalry] , [ville] -> ["town/city"] , [1er] -> [first] , 
[habitants] -> [inhabitants] , [nombre] -> [number] , [navire] -> [ship] , [membre] -> [member] , [équipage] -> [equipment] , 
[passager] -> [passenger] , [route] -> ["route/way"] , [pape] -> [Pope] , [symbole] -> [symbol] , [famille] -> [family] , 
[métaux] -> [metals] , [métal] -> [metal] , [affaiblissement] -> [weakening] , [façon] -> [fashion] , [prix] -> [prize] , [paix] -> [peace] , 
[yeux] -> [eyes] , [œil] -> [eye] , [prison] -> ["prison/imprisonment"] , [lutte] -> [struggle] , [concession] -> [compromise] , 
[idéaux] -> [ideals] , [idéal] -> [ideal] , [ministre] -> [minister] , [défense] -> [defense] , [accord] -> [agreement] , 
[stationnement] -> [parking] , [avion] -> [plane] , [appareil [0 | s]] -> [aircraft] , [semestre] -> ["half-of-the-year"] , 
[latéral [0 | s] " " gauche] -> left-side , [brésilien [0 | ne]] -> [Brazilian] , [roi] -> [king] , [hernie [0 | s] " " discale] -> ["slipped-disc"] , [hématome] -> [hematoma] , [trafic] -> [traffic] , [sécurités] -> [securities] , [sécurité] -> [security] , [rue] -> [road] , 
[poste] -> [post] , [accès] -> [access] , [journaliste] -> [journalist] , [féministe] -> [feminist] , [amoureuse | amoreux] -> [lover] , 
[musulman [e | 0]] -> [Muslim] , [fondamentaliste] -> [fundamentalist] , [lesbienne] -> [lesbian] , [effet] -> [effect] , 
[matière] -> [matters] , [diplomatie] -> [diplomacy] , [transparence] -> [transparency] , [liberté] -> [freedom] , [état] -> [state] , 
[cours] -> [courses] , [conseiller] -> [counselor] , [prises " " de " " décision] -> ["decision-making"] , [nerf] -> [nerve] , 
[anxiété] -> [anxiety] , [faveur] -> [favor] , [milliard] -> [billion], [monde] -> [world] , [presse] -> [press] , [jour] -> [day] , [ajustement] -> [adjustment] , [la " " maison] -> [home], [intellectuel] -> [intellectual] || [.#. | " "] _ ;

define Nouns subjectPronouns .o. properNouns .o. normalNouns;

#individual word translations - adjectives
define AdjectiveEndings [e | s | "es" | "ne" | "nes" | "le" | "les" | 0];

define Adjectives 
[distinct [AdjectiveEndings] ]-> [distinct] , [nucléaire [AdjectiveEndings] ]-> [nuclear] , [nippon [AdjectiveEndings] ] -> [Japanese] , [américain [AdjectiveEndings] ] -> [American] , [croissant [AdjectiveEndings] ] -> [increasing] , [comorien [AdjectiveEndings] ] -> [Comorian] , [somalien [AdjectiveEndings] ] -> [Somalian] , [ [neuf | neuve] [AdjectiveEndings] ]-> new , [naval [AdjectiveEndings] ] -> [naval] , [antipiraterie [AdjectiveEndings] ] -> [antipiracy] , [industriel [AdjectiveEndings]] -> [industrial] , [symbolique [AdjectiveEndings]] -> [symbolic] , [nobel [AdjectiveEndings] ] -> [Nobel] , [démocratique [AdjectiveEndings] ]-> [democratic] , [polonais [AdjectiveEndings]] -> [Polish] , [temporaire [AdjectiveEndings] ] -> [temporary] , [espagnol [AdjectiveEndings] ] -> [Spanish] , [madrilène [AdjectiveEndings]] -> [Madridian] , [normal [AdjectiveEndings] ] -> [normal] , [proche [AdjectiveEndings]]-> [close] , [chrétien [AdjectiveEndings] ] -> [Christian] , [[tous | tout] [AdjectiveEndings] ] -> [all] , [même [AdjectiveEndings] ] -> [same] , [publique [AdjectiveEndings]] -> [public] , [viable [AdjectiveEndings] ] -> [sustainable] , [malade [AdjectiveEndings] ] -> [sick] , [seul [AdjectiveEndings]] -> [alone] , [traditionnel [AdjectiveEndings]] -> [traditional] , [deux] -> [two] , [nécessaire [AdjectiveEndings]] -> [necessary] || [.#. | " "] _ [.#. | " "];

#individual word translations - adverbs
define Adverbs [apparemment] -> [apparently] , [déjà] -> [already] , [aussi] -> [also] , [désormais] -> [henceforth] , [également] -> [equally] , [quasiment] -> [almost] , [mortellement] -> [mortally] , [après] -> [after] , [gravement] -> [seriously] , [longtemps] -> ["a long time"] || [.#. | " "] _ [.#. | " "];


#the rest: determiners, complementizers, WH-words, prepositions, conjunctions
define finishing [ le | la | les ] -> [the] , [sa | son | ses] -> ["his/her/its"] , [leur | leurs] -> [their] , [un | une] -> ["a/an"] , [quelque] -> [some] , [ce | cette] -> [that] , [ces] -> ["these/those"] , [million " " de] -> [million] , [votre | vos] -> [your] , [par] -> [by] , [de] -> ["of/from"] , [dans] -> [in] , [jusque] -> [until] , [à] -> ["to/at"] , [pour] -> ["for/to"] , [en] -> ["in/to"] , [contre] -> [against] , [avec] -> [with] , [vers] -> [towards] , [sans] -> [without] , [sur] -> [on] , [entre] -> [between] , [depuis] -> [since] , 
[que] -> ["what/that/than"] , [qui] -> ["who/whom/that"] , [lorsque] -> [when] , [pourquoi] -> [why] , [où] -> [where] , [comment] -> [how] , [et] -> [and] , [ou] -> [or] , [mais] -> [but] || [.#. | " "] _ [.#. | " "];

define prepositions [by | "of/from" | in | until | "to/at" | "for/to" | "in/to" | against | with | towards | without | on | between | since | of | from | to | at | for];


#These are the functions directly called to translate sentences, in order
define preprocess toLowercase .o. removeContractions .o. punctuation1 .o. punctuation2 .o. punctuation3;
define replace Expressions .o. Verbs .o. Nouns .o. Adjectives .o. Adverbs .o. finishing;
define post [prepositions] -> 0 || [.#. | " " ] [prepositions | regarding "/" opposite] " " _ " " .o. " " -> 0 || " "_ .o. "the" -> "on" || _ " " [Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday] .o. [to " "] -> 0 || [could | must | she "/" her "/" it | he "/" it | we "/" us | one "/" you | they | it] " " _ .o. ["reach to/at"] -> [achieve] || [.#. | " "] _ [.#. | " "];

