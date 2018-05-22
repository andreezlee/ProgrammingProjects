import java.util.*;

public class Movement {
	
	static Set<String> Pronouns, Nouns, AuxiliaryVerbs, Verbs, Adjectives, Adverbs, Determiners, WH_words;

	public static void main(String[] args) {
		
		//creating sets for use
		String[] pronouns = {"she/her/it", "we/us", "he/it", "one/you", "all", "they", "it"};
		String[] properNouns = {"United-Nations", "France", "Youssouf-Bakayoko", "Wednesday", "Saturday", "April", "Washington",
				"Friday", "Cuidad-Juarez", "Aly-Zoulfecar", "Dar-Es-Salam", "Tanzania", "European-Union",
				"Operation-Atalanta", "Pope", "Sagrada-Familia", "Liu-Xiaobo", "Oslo", "China", "Bogdan-Klich",
				"Thursday", "F-16", "Hercules", "Warsaw", "Brazilian", "Marcelo", "Abdullah", "New-York", "November",
				"Mr.", "Deighton", "CIA", "Muslims", "WikiLeaks", "Cristina-Fernandez-de-Kirchner"};
		String[] singularNouns = {"candidate", "voice", "president", "commission", "evening", "result", "body", "man", "woman/wife",
				"victim", "date", "leader", "terrorism", "commitment", "euro", "currency", "yen", "night", "death",
				"citizen", "rivalry", "year", "town/city", "million", "inhabitant", "pirate", "number", "ship", "member",
				"passenger", "route/way", "force", "symbol", "family", "metal", "weakening", "dollar", "intellectual",
				"fashion", "prize", "struggle", "compromise", "ideal", "peace", "eye", "world", "prison/imprisonment",
				"minister", "defense", "agreement", "plane", "transport", "half-of-the-year", "club", "left", "king", "day",
				"slipped-disc", "hematoma", "traffic", "adjustment", "security", "road", "post", "access", "journalist",
				"agent", "feminist", "lover", "fundamentalist", "lesbian", "effect", "diplomacy", "transparency",
				"freedom", "expression", "time", "home", "anxiety", "balance", "favor", "stress", "attitude"};
		String[] pluralNouns = {"candidates", "voices", "presidents", "commissions", "evenings", "results", "bodies", "men", 
				"victims", "dates", "leaders", "nation(s)", "commitments", "euros", "currencies", "nights",
				"deaths", "citizens", "rivalries", "years", "towns/cities", "millions", "inhabitants", "pirates", "numbers",
				"ships", "members", "equipment", "passengers", "forces", "symbols", "metals", "dollars",
				"intellectuals", "prizes", "ideals", "eyes", "ministers", "agreements", "parking", "planes", "combat",
				"aircraft", "press", "clubs", "days", "adjustments", "roads", "posts", "police", "journalists", "feminists",
				"matters", "states", "finances", "courses", "nerves", "billions", "counselors", "decision-making"};
		Pronouns = new HashSet<String>();
		Nouns = new HashSet<String>();
		for(String s : pronouns) Pronouns.add(s);
		for(String s : properNouns) Nouns.add(s);
		for(String s : singularNouns) Nouns.add(s);
		for(String s : pluralNouns) Nouns.add(s);
		
		
		String[] auxiliaryVerbs = {"has", "had", "have", "be", "is", "could", "will", "was", "can", "must"};
		String[] normalVerbs = {"render", "fight", "compete", "pass/happen", "pursue", "rest", "do/make", "find", "stabilized", "reach",
				"address", "confirm", "take-possession", "blesses", "profited", "embodies", "search", "close", "deal-with", "achieve", "affect"};
		String[] participles = {"reiterated", "buried", "been", "recovered", "assassinated", "taken", "done/made",
				"detained", "did/made", "announced", "returned", "indicated", "found", "flown", "operated-on", 
				"complicated", "explained", "closed", "enraged", "tipped", "increasing", "forbidding", "going"};
		String[] adjectives = {"distinct", "nuclear", "Japanese", "American", "more", "Comorian", "Somalian", "new", "naval",
				"antipiracy", "traditional", "industrial", "symbolic", "Nobel", "democratic", "Polish", "temporary", "first",
				"Spanish", "Madridian", "two", "back-to-normal", "necessary", "Christian", "in-love", "at-the-same-time", 
				"same", "public", "sustainable", "sick", "long", "alone", "back-to-normal"};
		AuxiliaryVerbs = new HashSet<String>();
		Verbs = new HashSet<String>();
		Adjectives = new HashSet<String>();
		for(String s: auxiliaryVerbs){
			AuxiliaryVerbs.add(s);	Verbs.add(s);
		}for(String s: normalVerbs) Verbs.add(s);
		for(String s: participles){
			Verbs.add(s);	Adjectives.add(s);
		}for(String s: adjectives) Adjectives.add(s);
		
		String[] adverbsBefore = {"also", "henceforth", "equally", "almost", "mortally", "seriously"};
		Adverbs = new HashSet<String>();
		for(String s : adverbsBefore) Adverbs.add(s);
		
		String[] determiners = {"the", "his/her/its", "a/an", "some", "their", "that", "a", "your", "these/those"};
		Determiners = new HashSet<String>();
		for(String s : determiners) Determiners.add(s);
		
		String[] wh_words = {"what/that/than", "how", "where"};
		WH_words = new HashSet<String>();
		for(String s : wh_words) WH_words.add(s);
		
		
		//taking in user input and separating it into individual words
		ArrayList<Integer> indexes = new ArrayList<Integer>();
		ArrayList<String> words = new ArrayList<String>();
		Scanner reader = new Scanner(System.in);
		System.out.println("Enter a sentence:");
		String sentence = " " + reader.nextLine() + " ";
		for(int i = 0 ; i < sentence.length() ; i++){
			if(sentence.charAt(i) == ' ')
				indexes.add(i);
		}
		for(int i = 0 ; i < indexes.size() - 1; i++){
			words.add(sentence.substring(indexes.get(i) + 1, indexes.get(i + 1)));
		}
		adjectiveMovement(words);
		adverbMovement(words);
		objectPronounMovement(words);
		if(sentence.contains("?")){
			pronounDeletion(words);
			auxVerbMovement(words);
			doSupport(words);
		}
		System.out.println(postProcess(words));
	}
	
	//fixes up the sentence to make it presentable
	static String postProcess(ArrayList<String> strings){
		String output = "";
		output += strings.get(0).substring(0, 1).toUpperCase() + strings.get(0).substring(1);
		for(int i = 1; i < strings.size(); i++){
			if(strings.get(i).equals(",") || strings.get(i).equals(".") || strings.get(i).equals("?"))
				output += strings.get(i);
			else output += " " + strings.get(i);
		}
		return output.replace('-', ' ');
	}
	
	//swaps the positions of two words in the sentence
	static void swap(ArrayList<String> strings, int num1, int num2){
		String s = strings.get(num1);
		strings.set(num1, strings.get(num2));
		strings.set(num2, s);
	}
	
	//moves adjective to front of noun and makes the resulting adjective phrase become single entity
	static void adjectiveMovement(ArrayList<String> strings){
		int index = 0;
		while(index < strings.size()){
			if(index > 0 && Adjectives.contains(strings.get(index)) && Nouns.contains(strings.get(index - 1))){
				if(index == strings.size() - 1 || !(strings.get(index + 1).equals("by"))){
					strings.set(index - 1, strings.get(index) + " " + strings.get(index - 1));
					strings.remove(index);
					Nouns.add(strings.get(index - 1));
					index = 0;
				}
			}index++;
		}
	}
	
	//moves adverbs to the front of verbs, leaves those in front of adjectives as-is
	static void adverbMovement(ArrayList<String> strings){
		int adverbIndex = 0, verbIndex = 0;
		while(adverbIndex < strings.size()){
			if(Adverbs.contains(strings.get(adverbIndex))){
				if(adverbIndex == strings.size() - 1 || !(Adjectives.contains(strings.get(adverbIndex + 1)))){
					verbIndex = adverbIndex - 1;
					while(adverbIndex != 0 && verbIndex >= 0){
						if(Verbs.contains(strings.get(verbIndex))){
							strings.set(verbIndex, strings.get(adverbIndex) + " " + strings.get(verbIndex));
							strings.remove(adverbIndex);
							Nouns.add(strings.get(verbIndex));
							adverbIndex = 0;
						}verbIndex --;
					}
				}
			}adverbIndex++;
		}
	}
	
	//object pronouns behind a verb get moved to be after that verb
	static void objectPronounMovement(ArrayList<String> strings){
		int index = 0;
		while(index < strings.size() - 1){
			if(index > 0 && Pronouns.contains(strings.get(index)) && Verbs.contains(strings.get(index + 1))
					&& (Pronouns.contains(strings.get(index - 1)) || Nouns.contains(strings.get(index - 1)))){
				swap(strings, index, index + 1);
			}index++;
		}
	}

	//these functions will be used only with questions
	
	//in a question, removes placemarker pronouns that occur immediately after moved verbs
	static void pronounDeletion(ArrayList<String> strings){
		int nounIndex = 0, pronounIndex = 0;
		while(nounIndex < strings.size() - 1){
			if(Nouns.contains(strings.get(nounIndex)) && Verbs.contains(strings.get(nounIndex + 1))){
				pronounIndex = nounIndex + 1;
				while(pronounIndex < strings.size() - 1){
					if(Pronouns.contains(strings.get(pronounIndex)) && Verbs.contains(strings.get(pronounIndex - 1))){
						strings.remove(pronounIndex);
						return;
					}pronounIndex ++;
				}return;
			}nounIndex ++;
		}
	}
	
	//when necessary, moves the auxiliary verb in a question to the front of the question
	static void auxVerbMovement(ArrayList<String> strings){
		int index = 0;
		while(index < strings.size()){
			if(index > 0 && AuxiliaryVerbs.contains(strings.get(index)) && 
					(Pronouns.contains(strings.get(index - 1)) || Nouns.contains(strings.get(index - 1)))){
				String verb = strings.get(index);
				strings.remove(index);
				strings.add(0, verb);
				return;
			}			
			if(Verbs.contains(strings.get(index))){
				return;
			}index ++;
		}
	}
	
	//adds do-support after WH-words in questions
	static void doSupport(ArrayList<String> strings){
		int index = 0;
		while(index < strings.size() - 1){
			if(WH_words.contains(strings.get(index)) && (Nouns.contains(strings.get(index + 1)) 
					|| Pronouns.contains(strings.get(index + 1)) || Determiners.contains(strings.get(index + 1)))){
				strings.add(index + 1, "do/does");
				return;
			}
			if(WH_words.contains(strings.get(index)) && strings.get(index + 1).equals("to")){
				strings.set(index + 1, "do you");
				return;
			}
			if(WH_words.contains(strings.get(index))) return;
			index ++;
		}
	}
}
