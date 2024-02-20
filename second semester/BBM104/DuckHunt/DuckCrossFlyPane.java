import javafx.scene.layout.Pane;

import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.util.Duration;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
/**

The DuckCrossFlyPane class extends the JavaFX Pane class to create a pane
that displays an animated duck image that moves around the screen, and can
also fall when shot.
The challenging parts of the code are:
The implementation of the animation for the fall of the duck
The management of the duck's movement and collisions with the walls
*/
public class DuckCrossFlyPane extends Pane {
	private ImageView[] pngList = new ImageView[8];
	private String duckPath;
	ImageView duck;
	private int duckNo;
	private int reverseX = 1;
	private int reverseY = 1;
	private long x,y;
	private double dx = 15*DuckHunt.SCALE,dy = 15*DuckHunt.SCALE;
	private Timeline animation;
	Timeline animation2;
	MediaPlayer mediaPlayer = new MediaPlayer(
			new Media(getClass().getResource(
					"assets/effects/DuckFalls.mp3").toExternalForm()));;
	public boolean isDead = false;
	/**
	 * Constructs a new DuckCrossFlyPane with the specified parameters.
	 * 
	 * @param duckPath the path to the folder containing the duck image files
	 * @param lOrR a boolean indicating whether the duck should move left or right
	 * @param yCoord the initial y-coordinate of the duck
	 * @param uOrD a boolean indicating whether the duck should move up or down
	 * @param xCoord the initial x-coordinate of the duck
	 */
	DuckCrossFlyPane(String duckPath,boolean lOrR,long yCoord,boolean uOrD,long xCoord){
        mediaPlayer.setVolume(DuckHunt.VOLUME);

		this.isDead = false;
		this.setPrefHeight(256*DuckHunt.SCALE);
		this.setPrefWidth(240*DuckHunt.SCALE);
		this.duckNo = 0;
		this.duckPath = "assets/"+duckPath;
		for (int i = 1; i < 9; i++) {
			String tmp = String.valueOf(i);
			Image tmp2 = new Image(this.duckPath+tmp+".png");
			pngList[i-1] = new ImageView(tmp2);
			pngList[i-1].setFitHeight(tmp2.getHeight()*DuckHunt.SCALE);
			pngList[i-1].setFitWidth(tmp2.getWidth()*DuckHunt.SCALE);
			//Animation for the fall 
			pngList[i-1].setMouseTransparent(true);

		}

		
		
		//Initial state of coordinates and amount of pixel change

		if (lOrR) {
			reverseX = -reverseX;
			
		}
		if (uOrD) {
			reverseY = -reverseY;
		}
		this.x = xCoord;
		this.y = yCoord;
		
		
		
		//left and right animation
		animation = new Timeline(
			    new KeyFrame(Duration.ZERO, e -> moveDuck()),

			    new KeyFrame(Duration.millis(180))
			);
	    animation.setCycleCount(Timeline.INDEFINITE);
	    animation.play(); // Start animation
		

		
		//Animation of Death

		
	    animation2 = new Timeline(
	    		
	    	    new KeyFrame(Duration.millis(0), e -> fallImage()),
	    	    new KeyFrame(Duration.millis(300), e -> fall())
	    	);

	    
		

		}
		public void fall() {
			
			
			this.duck = pngList[7];
			this.getChildren().clear();
			
			duck.setLayoutX(x);
			duck.setLayoutY(y);

			duck.setScaleX(reverseX);
			duck.setScaleY(reverseY);

			this.getChildren().add(duck);
			
			
			//fall animation
			animation = new Timeline(
				    new KeyFrame(Duration.ZERO, e -> fallDuck()),
				    new KeyFrame(Duration.millis(500))
				);
		    animation.setCycleCount(15);
		    animation.play(); // Start animation


		    

		}
		public void fallDuck() {
			y+=35*DuckHunt.SCALE;
			duck.setLayoutX(x);
			duck.setLayoutY(y);
		}
		public void fallImage() {
		    mediaPlayer.play();
		    mediaPlayer.setOnEndOfMedia(() -> {
	            mediaPlayer.seek(Duration.ZERO);
	            mediaPlayer.stop();
	        });
			this.animation.stop();
			this.duck =pngList[6];
			this.getChildren().clear();
			
			duck.setLayoutX(x);
			duck.setLayoutY(y);
			duck.setScaleX(reverseX);
			duck.setScaleY(reverseY);
			this.getChildren().add(duck);
		}
		public void updateDuckNo() {
			if (duckNo==2) {
				duckNo = 0;
			}else {
				duckNo++;
			}
		}
		public void updateDuck() {
			
			this.duck = pngList[duckNo];
			this.getChildren().clear();
			this.getChildren().add(duck);
		}

		protected void moveDuck() {
		    updateDuckNo();
		    updateDuck();
		    
		    // Check for wall collisions
		    if (x <= 0) {
		        reverseX = Math.abs(reverseX);
		    } else if (x + duck.getFitWidth() >= 240*DuckHunt.SCALE) {
		        reverseX = -Math.abs(reverseX);
		    }
		    if (y <= 0) {
		        reverseY = Math.abs(reverseY);
		    } else if (y + duck.getFitHeight() >= 256*DuckHunt.SCALE) {
		        reverseY = -Math.abs(reverseY);
		    }

		    
			    // Update coordinates based on wall collisions
			    x += reverseX*dx;
			    y += reverseY*dy;
			    
			    // Update duck image view
			    duck.setLayoutX(x);
			    duck.setLayoutY(y);
		    

		    duck.setScaleX(reverseX);
		    duck.setScaleY(-reverseY);


		}

	

}
