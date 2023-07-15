(require [language.string str])

(def value "a")

(defn process-words
  [counts words]
  (for [word words]
    (if (not (str/blank? word))
      (let [cur-count (get counts word 0)]
        (assoc counts word (+ cur-count 1))))))
 
(defn process-line
  [counts line]
  (let [words (str/split (str/lower line))]
    (process-words counts words)))
