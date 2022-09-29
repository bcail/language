(println "Fahr    Celsius")
(loop
  [n 0]
  (do
    (print n)
    (print "    ")
    (println (/ (* 5 (- n 32)) 9))
    (if (< n 300)
      (recur (+ 20 n)))))
