(println "Fahr    Celsius")
(loop
  [n 0]
  (do
    (println (str n "    " (/ (* 5 (- n 32)) 9)))
    (if (< n 300)
      (recur (+ 20 n)))))
