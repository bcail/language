(defn process-words
  [counts words]
  (let [numwords (count words)]
    (loop [i 0]
      (if (= i numwords)
        nil 
        (let [word (nth words i)]
          (if (= false (str/blank? word))
            (let [curcount (get counts word 0)]
              (assoc counts word (+ curcount 1))))
          (recur (+ i 1)))))))
 
(defn process-line
  [counts line]
  (let [words (str/split (str/lower line))]
    (process-words counts words)))

(defn compare
  [a b]
  (> (nth a 1) (nth b 1)))

(let [counts {}]
  (loop [line (read-line)]
    (if (nil? line)
      nil
      (do
        (if (= false (str/blank? line))
          (process-line counts line))
        (recur (read-line)))))
  (let [sortedlist (sort compare (pairs counts))
        space " "]
    (for [entry sortedlist]
      (print (nth entry 0))
      (print space)
      (println (nth entry 1))))
