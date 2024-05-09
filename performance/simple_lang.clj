(require [language.string str])

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

(defn compare
  [a b]
  (> (nth a 1) (nth b 1)))

(let [counts {}]
  (loop [line (read-line)]
    (if (not (nil? line))
      (do
        (if (not (str/blank? line))
          (process-line counts line))
        (recur (read-line)))))
  (let [sortedlist (sort compare (pairs counts))
        space " "]
    (for [entry sortedlist]
      (print (nth entry 0))
      (print space)
      (println (nth entry 1)))))
